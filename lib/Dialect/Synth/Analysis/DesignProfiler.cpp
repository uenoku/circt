//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs early PPA estimation using LongestPathAnalysis.
// It classifies timing paths by clock domain and path type.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <numeric>

#define DEBUG_TYPE "synth-design-profiler"
using namespace circt;
using namespace synth;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_DESIGNPROFILER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

namespace {

/// Represents a clock domain identified by its Object (instance path + value).
/// Empty Object (null value) means "port" (unclocked).
using ClockDomain = Object;

struct Clock {
  Object root;
  virtual void print(llvm::raw_ostream &os = llvm::errs()) = 0;
};

using ClockPtr = std::unique_ptr<Clock>;

struct ClockGate : Clock {
  Clock *enable, *input;
};

struct ClockMux : Clock {
  Object sel;
  Clock *trueCase, *falseCase;
};

struct ClockDivider : Clock {
  Clock *input;
  int pow2;
};

struct ClockPort : Clock {};

struct ClockUnknown : Clock {
  Object input;
};

/// Path category based on start/end point types.
enum class PathCategory {
  RegToReg,   // Both endpoints are registers
  PortToReg,  // Input port to register
  RegToPort,  // Register to output port
  PortToPort, // Pure combinational
};

/// Key for grouping paths by clock domain pair.
struct ClockDomainPair {
  ClockDomain srcClock; // nullptr if from port
  ClockDomain dstClock; // nullptr if to port

  bool operator==(const ClockDomainPair &other) const {
    return srcClock == other.srcClock && dstClock == other.dstClock;
  }

  bool operator<(const ClockDomainPair &other) const {
    // Need strict weak ordering for map key if we use std::map with this struct
    // But we are using std::pair<std::string, std::string> as key in the map
    // below. So this operator might not be strictly needed unless we change the
    // map key. However, let's keep it simple.
    return false;
  }
};

/// Statistics for a group of paths.
struct PathStats {
  int64_t count = 0;
  int64_t maxDelay = 0;
  int64_t minDelay = std::numeric_limits<int64_t>::max();
  int64_t totalDelay = 0;
  std::vector<DataflowPath> topPaths;

  void addPath(const DataflowPath &path) {
    int64_t delay = path.getDelay();
    count++;
    maxDelay = std::max(maxDelay, delay);
    minDelay = std::min(minDelay, delay);
    totalDelay += delay;

    // Keep top 5 paths
    topPaths.push_back(path);
    std::sort(topPaths.begin(), topPaths.end(),
              [](const DataflowPath &a, const DataflowPath &b) {
                return a.getDelay() > b.getDelay();
              });
    if (topPaths.size() > 5)
      topPaths.resize(5);
  }

  int64_t avgDelay() const { return count > 0 ? totalDelay / count : 0; }
};

struct DesignProfilerPass
    : public impl::DesignProfilerBase<DesignProfilerPass> {
  using DesignProfilerBase::DesignProfilerBase;

  void runOnOperation() override;

private:
  SmallVector<Object> history;
  DenseMap<Object, ClockDomain> clockCache;
  /// Get the clock value for a path endpoint. Returns nullptr for ports.
  ClockDomain getClockForEndpoint(const Object &obj);

  /// Trace a clock signal back through the hierarchy if it is a port.
  ClockDomain traceClockSource(ClockDomain clkObj);

  /// Determine the path category based on start/end types.
  PathCategory categorize(const DataflowPath &path);

  /// Get a string name for a clock domain.
  std::string getClockName(ClockDomain clock);

  /// Write the timing report.
  LogicalResult writeReport(hw::HWModuleOp top, llvm::raw_ostream &os);

  /// Analysis results, populated during runOnOperation.
  const LongestPathAnalysis *analysis = nullptr;

  /// The module operation being analyzed.
  mlir::ModuleOp moduleOp;

  /// Instance graph for hierarchical traversal.
  circt::igraph::InstanceGraph *instanceGraph = nullptr;

  /// Instance path cache for manipulating instance paths.
  circt::igraph::InstancePathCache *pathCache = nullptr;
};

} // namespace

ClockDomain DesignProfilerPass::getClockForEndpoint(const Object &obj) {
  if (!obj.value)
    return {};

  Operation *defOp = obj.value.getDefiningOp();
  if (!defOp)
    return {}; // Block argument (port)

  Value clk = nullptr;
  // Check for register operations with clock.
  if (auto compreg = dyn_cast<seq::CompRegOp>(defOp))
    clk = compreg.getClk();
  else if (auto firreg = dyn_cast<seq::FirRegOp>(defOp))
    clk = firreg.getClk();

  if (clk) {
    // The clock value is defined in the same module as the register.
    // So we use the same instance path.
    Object clkObj(obj.instancePath, clk, 0);

    // // Verify and fix the instance path if needed.
    // // The instance path should point to the module containing the clock
    // value.
    // // If the leaf instance doesn't instantiate the clock's parent module,
    // // we need to drop the last instance from the path.
    // auto clkParentOp = llvm::dyn_cast<hw::HWModuleOp>(
    //     clk.getParentRegion()->getParentOp());
    // if (clkParentOp && !obj.instancePath.empty()) {
    //   auto instOp = dyn_cast<hw::InstanceOp>(obj.instancePath.leaf());
    //   if (instOp && instOp.getModuleNameAttr().getAttr() !=
    //   clkParentOp.getModuleNameAttr()) {
    //     // The instance path includes an extra instance - drop it
    //     clkObj = Object(obj.instancePath.dropBack(), clk, 0);
    //   }
    // }

    return traceClockSource(clkObj);
  }

  return {};
}
static void checkAssert(Value value) {
  if (!value)
    return;
  auto type = value.getType();
  if (!isa<seq::ClockType>(type)) {
    llvm::errs() << value << " is not a clock\n";
    assert(false);
  }
}

ClockDomain DesignProfilerPass::traceClockSource(ClockDomain clkObj) {
  auto it = clockCache.find(clkObj);
  if (it != clockCache.end())
    return it->second;
  history.clear();
  auto orig = clkObj;

  // Verify the instance path is consistent with the value's parent module.
  // This should have been fixed in getClockForEndpoint, but we check here
  // to catch any remaining issues.
  auto parentOp = llvm::dyn_cast<hw::HWModuleOp>(
      clkObj.value.getParentRegion()->getParentOp());
  if (parentOp && !clkObj.instancePath.empty()) {
    auto instOp = dyn_cast<hw::InstanceOp>(clkObj.instancePath.leaf());
    if (instOp &&
        instOp.getModuleNameAttr().getAttr() != parentOp.getModuleNameAttr()) {
      llvm::errs() << "ERROR: Instance path is still incorrect after fix!\n";
      llvm::errs() << "Instance path: " << clkObj.instancePath << "\n";
      llvm::errs() << "Parent op: " << parentOp.getModuleName() << "\n";
      llvm::errs() << "Instance op: " << instOp << "\n";
      assert(false && "Instance path consistency check failed");
    }
  }
  while (true) {
    if (!clkObj.value)
      break;

    history.push_back(clkObj);
    if (!isa<seq::ClockType>(clkObj.value.getType())) {
      for (auto obj : history) {
        obj.print(llvm::errs());
        llvm::errs() << "\n";
      }
      checkAssert(clkObj.value);
    }

    // If defined by op, it's a local source.
    if (auto *op = clkObj.value.getDefiningOp()) {
      if (auto wire = dyn_cast<hw::WireOp>(op)) {
        clkObj = Object(clkObj.instancePath, wire.getInput(), 0);
        continue;
      }

      if (auto clockDiv = dyn_cast<seq::ClockGateOp>(op)) {
        clkObj = Object(clkObj.instancePath, clockDiv.getInput(), 0);
        continue;
      }

      if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
        // The clock comes from an instance output. We need to trace into the
        // instance to find where this output comes from inside the instance.

        // If we don't have an instance graph or path cache, we can't trace into
        // instances.
        if (!instanceGraph || !pathCache)
          break;

        // Get the result number to know which output port this is.
        auto resultValue = dyn_cast<OpResult>(clkObj.value);
        if (!resultValue)
          break;
        unsigned resultNum = resultValue.getResultNumber();

        // Look up the referenced module.
        auto moduleName = inst.getReferencedModuleNameAttr();
        auto *moduleNode = instanceGraph->lookup(moduleName);
        if (!moduleNode)
          break;
        auto hwModule =
            dyn_cast<hw::HWModuleOp>(*moduleNode->getModule().getOperation());
        if (!hwModule)
          break;

        // Get the output operation and find the corresponding operand.
        auto *terminator = hwModule.getBodyBlock()->getTerminator();
        assert(resultNum < terminator->getNumOperands());
        Value outputVal = terminator->getOperand(resultNum);

        // Only continue tracing if the output value is still a clock type.
        // If the module converts clock to data, we stop here.
        if (!isa<seq::ClockType>(outputVal.getType()))
          break;

        // Build a new instance path by appending this instance.
        // We're going INTO the instance, so we append it to the path.
        auto newPath = pathCache->appendInstance(clkObj.instancePath, inst);

        clkObj = Object(newPath, outputVal, 0);
        continue;
      }
      break;
    }

    // It's a BlockArgument.
    auto blockArg = dyn_cast<BlockArgument>(clkObj.value);
    if (!blockArg)
      break;

    // Check if it's a module port.
    auto *ownerBlock = blockArg.getOwner();
    if (!isa<hw::HWModuleOp>(ownerBlock->getParentOp()))
      break;

    // If we are at the top of the instance path, we can't go up.
    if (clkObj.instancePath.empty())
      break;

    // Get the instance that instantiated this module.
    auto instOp = clkObj.instancePath.leaf();

    // Handle hw::InstanceOp.
    if (auto hwInst = dyn_cast<hw::InstanceOp>(instOp.getOperation())) {
      unsigned argIdx = blockArg.getArgNumber();

      // Verify the instance actually instantiates the module that owns the
      // block arg
      auto ownerMod = dyn_cast<hw::HWModuleOp>(ownerBlock->getParentOp());
      // if (ownerMod && hwInst.getModuleName() != ownerMod.getModuleName()) {
      //   llvm::errs() << "ERROR: Instance " << hwInst.getInstanceName()
      //                << " references module " << hwInst.getModuleName()
      //                << " but BlockArg is from module "
      //                << ownerMod.getModuleName() << "\n";
      //   llvm::errs() << "This means the instance path is incorrect!\n";
      //   break;
      // }

      if (argIdx >= hwInst.getNumOperands()) {
        llvm::errs() << "DEBUG: BlockArg " << blockArg << " at index " << argIdx
                     << ", instance " << hwInst.getInstanceName()
                     << " (module: " << hwInst.getModuleName() << ")" << " has "
                     << hwInst.getNumOperands() << " operands\n";
        llvm::errs() << "DEBUG: BlockArg owner module: ";
        if (auto mod = dyn_cast<hw::HWModuleOp>(ownerBlock->getParentOp()))
          llvm::errs() << mod.getModuleName();
        llvm::errs() << "\n";
      }
      assert(argIdx < hwInst.getNumOperands());
      Value parentVal = hwInst.getInputs()[argIdx];
      // llvm::errs() << "DEBUG: Got operand " << parentVal << "\n";
      clkObj = Object(clkObj.instancePath.dropBack(), parentVal, 0);
      continue;
    }

    // Stop if we can't handle the instance type.
    break;
  }
  clockCache[orig] = clkObj;
  return clkObj;
}

PathCategory DesignProfilerPass::categorize(const DataflowPath &path) {
  bool startIsReg = false;
  bool endIsReg = false;

  // Check start point.
  const Object &start = path.getStartPoint();
  if (start.value && start.value.getDefiningOp()) {
    Operation *op = start.value.getDefiningOp();
    startIsReg = isa<seq::CompRegOp, seq::FirRegOp>(op);
  }

  // Check end point.
  if (auto *obj = std::get_if<Object>(&path.getEndPoint())) {
    if (obj->value && obj->value.getDefiningOp()) {
      Operation *op = obj->value.getDefiningOp();
      endIsReg = isa<seq::CompRegOp, seq::FirRegOp>(op);
    }
  }
  // OutputPort endpoint means end is a port.

  if (startIsReg && endIsReg)
    return PathCategory::RegToReg;
  if (!startIsReg && endIsReg)
    return PathCategory::PortToReg;
  if (startIsReg && !endIsReg)
    return PathCategory::RegToPort;
  return PathCategory::PortToPort;
}

std::string DesignProfilerPass::getClockName(ClockDomain clock) {
  if (!clock.value)
    return "(port)";

  std::string name;
  // Try to get a meaningful name.
  if (auto blockArg = dyn_cast<BlockArgument>(clock.value)) {
    if (auto hwMod =
            dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp())) {
      name = hwMod.getInputName(blockArg.getArgNumber()).str();
    } else {
      name = "arg" + std::to_string(blockArg.getArgNumber());
    }
  } else if (Operation *defOp = clock.value.getDefiningOp()) {
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(defOp)) {
      name = instanceOp.getInstanceName().str();
      // Add a port name.
      if (auto portName = instanceOp.getOutputName(
              cast<OpResult>(clock.value).getResultNumber())) {
        name += "/" + portName.getValue().str();
      }
    } else if (auto attr = defOp->getAttrOfType<StringAttr>("sv.namehint"))
      name = attr.getValue().str();
    else if (auto attr = defOp->getAttrOfType<StringAttr>("name"))
      name = attr.getValue().str();
    else
      name = defOp->getName().getStringRef().str();
  } else {
    name = "(unknown)";
  }

  // Prepend instance path.
  std::string pathStr;
  llvm::raw_string_ostream ss(pathStr);
  if (!clock.instancePath.empty()) {
    for (auto inst : clock.instancePath) {
      ss << inst.getInstanceName() << "/";
    }
  }
  return ss.str() + name;
}

LogicalResult DesignProfilerPass::writeReport(hw::HWModuleOp top,
                                              llvm::raw_ostream &os) {
  auto moduleName = top.getModuleNameAttr();

  // Collect all paths.
  SmallVector<DataflowPath> allPaths;
  if (failed(analysis->getAllPaths(moduleName, allPaths, /*elaborate=*/true))) {
    return failure();
  }

  llvm::dbgs() << "Found " << allPaths.size() << " paths\n";

  // Group paths by category and clock domain.
  using ClockPairStats =
      std::map<std::pair<std::string, std::string>, PathStats>;
  ClockPairStats regToRegStats;
  std::map<std::string, PathStats> portToRegStats;
  std::map<std::string, PathStats> regToPortStats;
  PathStats portToPortStats;

  int64_t totalMaxDelay = 0;
  llvm::DenseSet<Object> uniqueClocks;

  for (auto &path : allPaths) {
    int64_t delay = path.getDelay();
    totalMaxDelay = std::max(totalMaxDelay, delay);

    PathCategory cat = categorize(path);
    ClockDomain srcClock = getClockForEndpoint(path.getStartPoint());
    ClockDomain dstClock = {};

    if (auto *obj = std::get_if<Object>(&path.getEndPoint()))
      dstClock = getClockForEndpoint(*obj);

    if (srcClock.value)
      uniqueClocks.insert(srcClock);
    if (dstClock.value)
      uniqueClocks.insert(dstClock);

    switch (cat) {
    case PathCategory::RegToReg: {
      auto key = std::make_pair(getClockName(srcClock), getClockName(dstClock));
      regToRegStats[key].addPath(path);
      break;
    }
    case PathCategory::PortToReg:
      portToRegStats[getClockName(dstClock)].addPath(path);
      break;
    case PathCategory::RegToPort:
      regToPortStats[getClockName(srcClock)].addPath(path);
      break;
    case PathCategory::PortToPort:
      portToPortStats.addPath(path);
      break;
    }
  }

  auto printStats = [&](const PathStats &stats) {
    os << llvm::format("  Paths: %d | Max: %d | Min: %d | Avg: %d\n",
                       stats.count, stats.maxDelay, stats.minDelay,
                       stats.avgDelay());
    os << "  Top 5 Critical Paths:\n";
    for (const auto &path : stats.topPaths) {
      os << "    [delay=" << path.getDelay() << "] ";
      path.getStartPoint().print(os);
      os << " -> ";
      // We need to cast away constness to call printEndPoint because it takes
      // non-const reference in current API, or use a workaround.
      // Checking DataflowPath API: void printEndPoint(llvm::raw_ostream &os);
      // It is not const.
      const_cast<DataflowPath &>(path).printEndPoint(os);
      os << "\n";
    }
  };

  // Write report.
  os << "# Design Profile: " << moduleName.getValue() << "\n\n";

  os << "## Summary\n";
  os << "Total paths: " << allPaths.size();
  os << " | Max delay: " << totalMaxDelay;
  os << " | Clock domains: " << uniqueClocks.size() << "\n\n";

  os << "## Clock Domains\n";
  for (auto clock : uniqueClocks) {
    os << "  " << getClockName(clock) << "\n";
  }
  os << "\n";

  // Reg-to-Reg.
  if (!regToRegStats.empty()) {
    os << "## Reg-to-Reg Paths\n";
    for (auto &[clockPair, stats] : regToRegStats) {
      os << "### " << clockPair.first << " -> " << clockPair.second << "\n";
      printStats(stats);
    }
    os << "\n";
  }

  // Port-to-Reg.
  if (!portToRegStats.empty()) {
    os << "## Port-to-Reg Paths\n";
    for (auto &[clock, stats] : portToRegStats) {
      os << "### -> " << clock << "\n";
      printStats(stats);
    }
    os << "\n";
  }

  // Reg-to-Port.
  if (!regToPortStats.empty()) {
    os << "## Reg-to-Port Paths\n";
    for (auto &[clock, stats] : regToPortStats) {
      os << "### " << clock << " ->\n";
      printStats(stats);
    }
    os << "\n";
  }

  // Port-to-Port.
  if (portToPortStats.count > 0) {
    os << "## Port-to-Port Paths\n";
    printStats(portToPortStats);
    os << "\n";
  }

  return success();
}

void DesignProfilerPass::runOnOperation() {
  auto module = getOperation();
  moduleOp = module;

  // Validate top module name.
  if (topModuleName.empty()) {
    module.emitError() << "top module name must be specified";
    return signalPassFailure();
  }

  auto topNameAttr = StringAttr::get(&getContext(), topModuleName);

  // Create analysis.
  auto am = getAnalysisManager();
  LongestPathAnalysis localAnalysis(
      module, am,
      LongestPathAnalysisOptions(/*lazyComputation=*/false,
                                 /*keepOnlyMaxDelayPaths=*/true, topNameAttr));
  analysis = &localAnalysis;

  // Get instance graph and create path cache.
  instanceGraph = &am.getAnalysis<circt::igraph::InstanceGraph>();
  circt::igraph::InstancePathCache localPathCache(*instanceGraph);
  pathCache = &localPathCache;

  // Find the top module.
  hw::HWModuleOp topModule;
  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    if (hwMod.getModuleNameAttr() == topNameAttr) {
      topModule = hwMod;
      break;
    }
  }

  if (!topModule) {
    module.emitError() << "top module '" << topModuleName << "' not found";
    return signalPassFailure();
  }

  // Create output directory: <reportDir>/<topModuleName>/
  SmallString<256> outputDir(reportDir);
  llvm::sys::path::append(outputDir, topModuleName);

  if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
    module.emitError() << "failed to create report directory: " << ec.message();
    return signalPassFailure();
  }

  // Write timing report.
  SmallString<256> timingPath(outputDir);
  llvm::sys::path::append(timingPath, "timing.txt");

  std::string error;
  auto file = mlir::openOutputFile(timingPath, &error);
  if (!file) {
    module.emitError() << "failed to open output file: " << error;
    return signalPassFailure();
  }

  if (failed(writeReport(topModule, file->os())))
    return signalPassFailure();

  file->keep();
  markAllAnalysesPreserved();
}
