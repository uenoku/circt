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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"
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
  Object value;
  virtual ~Clock() = default;
  virtual void print(llvm::raw_ostream &os = llvm::errs()) = 0;
};

struct ClockGate : Clock {
  Object enable;
  Clock *input;

  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockGate(enable=";
    enable.print(os);
    os << ", input=";
    if (input)
      input->print(os);
    else
      os << "null";
    os << ")";
  }
};

struct ClockMux : Clock {
  Object sel;
  Clock *trueCase, *falseCase;

  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockMux(sel=";
    sel.print(os);
    os << ", true=";
    if (trueCase)
      trueCase->print(os);
    else
      os << "null";
    os << ", false=";
    if (falseCase)
      falseCase->print(os);
    else
      os << "null";
    os << ")";
  }
};

struct ClockDivider : Clock {
  Clock *input;
  int pow2;

  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockDivider(pow2=" << pow2 << ", input=";
    if (input)
      input->print(os);
    else
      os << "null";
    os << ")";
  }
};

struct ClockInverter : Clock {
  Clock *input;

  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockInverter(input=";
    if (input)
      input->print(os);
    else
      os << "null";
    os << ")";
  }
};

struct ClockPort : Clock {
  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockPort(";
    value.print(os);
    os << ")";
  }
};

struct ClockUnknown : Clock {
  void print(llvm::raw_ostream &os = llvm::errs()) override {
    os << "ClockUnknown(";
    value.print(os);
    os << ")";
  }
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
  DenseMap<Object, Clock *> clockTreeCache;
  SmallVector<Clock *> allocatedClocks;

  // Helper to allocate and track a clock object
  template <typename T>
  T *allocateClock() {
    T *clk = new T();
    allocatedClocks.push_back(clk);
    return clk;
  }

  // Cleanup allocated clocks
  void cleanupClocks() {
    for (auto *clk : allocatedClocks) {
      delete clk;
    }
    allocatedClocks.clear();
    clockTreeCache.clear();
  }

  /// Get the clock value for a path endpoint. Returns nullptr for ports.
  ClockDomain getClockForEndpoint(const Object &obj);

  /// Trace a clock signal back through the hierarchy if it is a port.
  ClockDomain traceClockSource(ClockDomain clkObj);

  /// Trace a clock signal and build a Clock tree structure.
  Clock *traceClockTree(Object clkObj);

  /// Print a clock tree in a hierarchical format.
  void printClockTree(Clock *clk, llvm::raw_ostream &os,
                      const std::string &indent = "", bool isLast = true);

  /// Find all clock gates in a clock tree.
  void findClockGates(Clock *clk, SmallVectorImpl<ClockGate *> &gates);

  /// Check if two clock trees have a common clock gate input.
  bool haveCommonClockGate(Clock *clk1, Clock *clk2);

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
    // if (!isa<seq::ClockType>(clkObj.value.getType())) {
    //   for (auto obj : history) {
    //     obj.print(llvm::errs());
    //     llvm::errs() << "\n";
    //   }
    //   checkAssert(clkObj.value);
    // }

    // If defined by op, it's a local source.
    if (auto *op = clkObj.value.getDefiningOp()) {
      if (auto wire = dyn_cast<hw::WireOp>(op)) {
        clkObj = Object(clkObj.instancePath, wire.getInput(), 0);
        continue;
      }

      // Handle clock gate - trace through to the input clock
      if (auto clockGate = dyn_cast<seq::ClockGateOp>(op)) {
        clkObj = Object(clkObj.instancePath, clockGate.getInput(), 0);
        continue;
      }
      // Handle clock gate - trace through to the input clock
      if (auto fromClock = dyn_cast<seq::FromClockOp>(op)) {
        clkObj = Object(clkObj.instancePath, fromClock.getInput(), 0);
        continue;
      }

      if (auto toClock = dyn_cast<seq::ToClockOp>(op)) {
        clkObj = Object(clkObj.instancePath, toClock.getInput(), 0);
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

        // // Only continue tracing if the output value is still a clock type.
        // // If the module converts clock to data, we stop here.
        // if (!isa<seq::ClockType>(outputVal.getType()))
        //   break;

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

Clock *DesignProfilerPass::traceClockTree(Object clkObj) {
  // Check cache first
  auto it = clockTreeCache.find(clkObj);
  if (it != clockTreeCache.end())
    return it->second;

  if (!clkObj.value) {
    // Null clock - return ClockUnknown
    auto *clk = allocateClock<ClockUnknown>();
    clk->value = clkObj;
    clockTreeCache[clkObj] = clk;
    return clk;
  }

  // If defined by op, build the appropriate Clock structure
  if (auto *op = clkObj.value.getDefiningOp()) {
    // Handle wire - trace through
    if (auto wire = dyn_cast<hw::WireOp>(op)) {
      Object inputObj(clkObj.instancePath, wire.getInput(), 0);
      return traceClockTree(inputObj);
    }

    // Handle clock gate
    if (auto clockGate = dyn_cast<seq::ClockGateOp>(op)) {
      auto *clkGate = allocateClock<ClockGate>();
      clkGate->value = clkObj;
      clkGate->enable = Object(clkObj.instancePath, clockGate.getEnable(), 0);
      Object inputObj(clkObj.instancePath, clockGate.getInput(), 0);
      clkGate->input = traceClockTree(inputObj);
      clockTreeCache[clkObj] = clkGate;
      return clkGate;
    }

    // Handle clock mux
    if (auto clockMux = dyn_cast<seq::ClockMuxOp>(op)) {
      auto *clkMux = allocateClock<ClockMux>();
      clkMux->value = clkObj;
      clkMux->sel = Object(clkObj.instancePath, clockMux.getCond(), 0);
      Object trueObj(clkObj.instancePath, clockMux.getTrueClock(), 0);
      Object falseObj(clkObj.instancePath, clockMux.getFalseClock(), 0);
      clkMux->trueCase = traceClockTree(trueObj);
      clkMux->falseCase = traceClockTree(falseObj);
      clockTreeCache[clkObj] = clkMux;
      return clkMux;
    }

    // Handle clock divider
    if (auto clockDiv = dyn_cast<seq::ClockDividerOp>(op)) {
      auto *clkDivider = allocateClock<ClockDivider>();
      clkDivider->value = clkObj;
      clkDivider->pow2 = clockDiv.getPow2();
      Object inputObj(clkObj.instancePath, clockDiv.getInput(), 0);
      clkDivider->input = traceClockTree(inputObj);
      clockTreeCache[clkObj] = clkDivider;
      return clkDivider;
    }

    // Handle clock inverter
    if (auto clockInv = dyn_cast<seq::ClockInverterOp>(op)) {
      auto *clkInverter = allocateClock<ClockInverter>();
      clkInverter->value = clkObj;
      Object inputObj(clkObj.instancePath, clockInv.getInput(), 0);
      clkInverter->input = traceClockTree(inputObj);
      clockTreeCache[clkObj] = clkInverter;
      return clkInverter;
    }

    // Handle seq::ToClockOp - trace through transparently
    if (auto toClock = dyn_cast<seq::ToClockOp>(op)) {
      Object inputObj(clkObj.instancePath, toClock.getInput(), 0);
      return traceClockTree(inputObj);
    }

    // Handle seq::FromClockOp - trace through transparently
    if (auto fromClock = dyn_cast<seq::FromClockOp>(op)) {
      Object inputObj(clkObj.instancePath, fromClock.getInput(), 0);
      return traceClockTree(inputObj);
    }

    // Handle synth::aig::AndInverterOp with single inverted operand (NOT gate)
    if (auto inverter = dyn_cast<synth::aig::AndInverterOp>(op)) {
      if (inverter.getNumOperands() == 1 && inverter.isInverted(0)) {
        auto *clkInverter = allocateClock<ClockInverter>();
        clkInverter->value = clkObj;
        Object inputObj(clkObj.instancePath, inverter.getOperand(0), 0);
        clkInverter->input = traceClockTree(inputObj);
        clockTreeCache[clkObj] = clkInverter;
        return clkInverter;
      }
    }

    // Handle instance output - trace into the instance
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      if (instanceGraph && pathCache) {
        auto resultValue = dyn_cast<OpResult>(clkObj.value);
        if (resultValue) {
          unsigned resultNum = resultValue.getResultNumber();
          auto moduleName = inst.getReferencedModuleNameAttr();
          auto *moduleNode = instanceGraph->lookup(moduleName);
          if (moduleNode) {
            auto hwModule = dyn_cast<hw::HWModuleOp>(
                *moduleNode->getModule().getOperation());
            if (hwModule) {
              auto *terminator = hwModule.getBodyBlock()->getTerminator();
              Value outputVal = terminator->getOperand(resultNum);
              auto newPath =
                  pathCache->appendInstance(clkObj.instancePath, inst);
              Object innerObj(newPath, outputVal, 0);
              return traceClockTree(innerObj);
            }
          }
        }
      }
    }

    // Unknown operation - return ClockUnknown
    auto *clk = allocateClock<ClockUnknown>();
    clk->value = clkObj;
    clockTreeCache[clkObj] = clk;
    return clk;
  }

  // It's a BlockArgument (port)
  auto blockArg = dyn_cast<BlockArgument>(clkObj.value);
  if (blockArg) {
    auto *ownerBlock = blockArg.getOwner();
    if (isa<hw::HWModuleOp>(ownerBlock->getParentOp())) {
      // If we can trace up through the instance hierarchy, do so
      if (!clkObj.instancePath.empty()) {
        auto instOp = clkObj.instancePath.leaf();
        if (auto hwInst = dyn_cast<hw::InstanceOp>(instOp.getOperation())) {
          unsigned argIdx = blockArg.getArgNumber();
          if (argIdx < hwInst.getNumOperands()) {
            Value parentVal = hwInst.getInputs()[argIdx];
            Object parentObj(clkObj.instancePath.dropBack(), parentVal, 0);
            return traceClockTree(parentObj);
          }
        }
      }

      // It's a top-level port
      auto *clk = allocateClock<ClockPort>();
      clk->value = clkObj;
      clockTreeCache[clkObj] = clk;
      return clk;
    }
  }

  // Unknown - return ClockUnknown
  auto *clk = allocateClock<ClockUnknown>();
  clk->value = clkObj;
  clockTreeCache[clkObj] = clk;
  return clk;
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

void DesignProfilerPass::printClockTree(Clock *clk, llvm::raw_ostream &os,
                                        const std::string &indent,
                                        bool isLast) {
  if (!clk) {
    os << indent << (isLast ? "└── " : "├── ") << "(null)\n";
    return;
  }

  std::string branch = isLast ? "└── " : "├── ";
  std::string childIndent = indent + (isLast ? "    " : "│   ");

  if (auto *gate = dynamic_cast<ClockGate *>(clk)) {
    os << indent << branch << "ClockGate\n";
    os << childIndent << "├── enable: ";
    gate->enable.print(os);
    os << "\n";
    os << childIndent << "└── input:\n";
    printClockTree(gate->input, os, childIndent + "    ", true);
  } else if (auto *mux = dynamic_cast<ClockMux *>(clk)) {
    os << indent << branch << "ClockMux\n";
    os << childIndent << "├── selector: ";
    mux->sel.print(os);
    os << "\n";
    os << childIndent << "├── true:\n";
    printClockTree(mux->trueCase, os, childIndent + "│   ", false);
    os << childIndent << "└── false:\n";
    printClockTree(mux->falseCase, os, childIndent + "    ", true);
  } else if (auto *div = dynamic_cast<ClockDivider *>(clk)) {
    os << indent << branch << "ClockDivider (÷" << (1 << div->pow2) << ")\n";
    os << childIndent << "└── input:\n";
    printClockTree(div->input, os, childIndent + "    ", true);
  } else if (auto *inv = dynamic_cast<ClockInverter *>(clk)) {
    os << indent << branch << "ClockInverter\n";
    os << childIndent << "└── input:\n";
    printClockTree(inv->input, os, childIndent + "    ", true);
  } else if (auto *port = dynamic_cast<ClockPort *>(clk)) {
    os << indent << branch << "ClockPort: ";
    port->value.print(os);
    os << "\n";
  } else if (auto *unknown = dynamic_cast<ClockUnknown *>(clk)) {
    os << indent << branch << "ClockUnknown: ";
    unknown->value.print(os);
    os << "\n";
  } else {
    os << indent << branch << "(unknown clock type)\n";
  }
}

void DesignProfilerPass::findClockGates(Clock *clk,
                                        SmallVectorImpl<ClockGate *> &gates) {
  if (!clk)
    return;

  if (auto *gate = dynamic_cast<ClockGate *>(clk)) {
    gates.push_back(gate);
    findClockGates(gate->input, gates);
  } else if (auto *mux = dynamic_cast<ClockMux *>(clk)) {
    findClockGates(mux->trueCase, gates);
    findClockGates(mux->falseCase, gates);
  } else if (auto *div = dynamic_cast<ClockDivider *>(clk)) {
    findClockGates(div->input, gates);
  } else if (auto *inv = dynamic_cast<ClockInverter *>(clk)) {
    findClockGates(inv->input, gates);
  }
  // ClockPort and ClockUnknown are leaf nodes, no recursion needed
}

bool DesignProfilerPass::haveCommonClockGate(Clock *clk1, Clock *clk2) {
  if (!clk1 || !clk2)
    return false;

  SmallVector<ClockGate *, 8> gates1, gates2;
  findClockGates(clk1, gates1);
  findClockGates(clk2, gates2);

  // Check if any clock gates have the same enable signal
  for (auto *gate1 : gates1) {
    for (auto *gate2 : gates2) {
      // Compare enable signals
      if (gate1->enable == gate2->enable) {
        return true;
      }
    }
  }

  return false;
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
      DenseMap<std::pair<ClockDomain, ClockDomain>, PathStats>;
  ClockPairStats regToRegStats;
  DenseMap<ClockDomain, PathStats> portToRegStats;
  DenseMap<ClockDomain, PathStats> regToPortStats;
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
      auto key = std::make_pair(srcClock, dstClock);
      regToRegStats[key].addPath(path);
      break;
    }
    case PathCategory::PortToReg:
      portToRegStats[dstClock].addPath(path);
      break;
    case PathCategory::RegToPort:
      regToPortStats[srcClock].addPath(path);
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
      // Print start point with _reg suffix for registers, no location (shown in
      // path details)
      path.getStartPoint().print(os, /*withLoc=*/false, /*addRegSuffix=*/true);
      os << " -> ";
      // We need to cast away constness to call printEndPoint because it takes
      // non-const reference in current API, or use a workaround.
      // Checking DataflowPath API: void printEndPoint(llvm::raw_ostream &os);
      // It is not const.
      const_cast<DataflowPath &>(path).printEndPoint(os, /*withLoc=*/false);
      os << "\n";

      // Reconstruct and print intermediate path points with locations
      SmallVector<DebugPoint> pathHistory;
      if (succeeded(analysis->reconstructPath(path, pathHistory))) {
        if (pathHistory.size() >= 1) {
          os << "      Path details:\n";
          for (size_t i = 0; i < pathHistory.size(); ++i) {
            const auto &point = pathHistory[i];
            os << "        [" << i << "] ";
            // Don't show location for start/end points (first and last), show
            // for intermediate points. Add _reg suffix for start/end points.
            bool isStartOrEnd = (i == 0 || i == pathHistory.size() - 1);
            point.print(os, /*withLoc=*/!isStartOrEnd,
                        /*addRegSuffix=*/isStartOrEnd);
            os << "\n";
          }
        }
      }
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
    os << "### " << getClockName(clock) << "\n";
    Clock *clockTree = traceClockTree(clock);
    if (clockTree) {
      printClockTree(clockTree, os, "", true);
    } else {
      os << "  (no clock tree available)\n";
    }
    os << "\n";
  }
  os << "\n";

  // Reg-to-Reg.
  if (!regToRegStats.empty()) {
    os << "## Reg-to-Reg Paths\n";
    for (auto &[clockPair, stats] : regToRegStats) {
      std::string srcName = getClockName(clockPair.first);
      std::string dstName = getClockName(clockPair.second);
      os << "### " << srcName << " -> " << dstName << "\n";

      // Check for common clock gate
      Clock *srcClockTree = traceClockTree(clockPair.first);
      Clock *dstClockTree = traceClockTree(clockPair.second);

      if (srcClockTree && dstClockTree) {
        if (haveCommonClockGate(srcClockTree, dstClockTree)) {
          os << "  ⚠️  Common clock gate detected between source and destination clocks\n";
        }
      }

      printStats(stats);
    }
    os << "\n";
  }

  // Port-to-Reg.
  if (!portToRegStats.empty()) {
    os << "## Port-to-Reg Paths\n";
    for (auto &[clock, stats] : portToRegStats) {
      std::string clockName = getClockName(clock);
      os << "### -> " << clockName << "\n";
      printStats(stats);
    }
    os << "\n";
  }

  // Reg-to-Port.
  if (!regToPortStats.empty()) {
    os << "## Reg-to-Port Paths\n";
    for (auto &[clock, stats] : regToPortStats) {
      std::string clockName = getClockName(clock);
      os << "### " << clockName << " ->\n";
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

  if (failed(writeReport(topModule, file->os()))) {
    cleanupClocks();
    return signalPassFailure();
  }

  file->keep();
  cleanupClocks();
  markAllAnalysesPreserved();
}
