//===- EmitSDC.cpp - Emit SDC timing constraints ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Discovers flop clock roots by tracing clock pins (liberty `clock = "true"`
// input pins) back through single-input cells and hierarchy to top-level
// input ports, then emits `create_clock` constraints plus default
// input/output delays. Clocks that do not resolve to a top port
// (comb-generated, tied, or lost in hierarchy) are reported as
// unconstrained comments for the engineer to disposition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ToolOutputFile.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_EMITSDC
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-emit-sdc"

namespace {

// Outcome of resolving one flop clock net.
struct ClockResolution {
  SmallVector<std::string> roots;
  SmallVector<std::string> sources;
  // True when the trace passed through a module instantiated more than
  // once: the roots are a union over instantiation contexts, so per-flop
  // attribution needs hierarchical scoping (future work).
  bool sharedContext = false;
  // Non-empty when no top-port root was found.
  std::string reason;
};

/// A named clock-gate rule: output port `out` of cells named `cell` traces
/// to input port `in` for domain attribution (gate ignored, standard for
/// first-order SDC; gating checks are out of scope).
struct GateRule {
  std::string cell;
  std::string out;
  std::string in;
};

struct EmitSDCPass : public circt::synth::impl::EmitSDCBase<EmitSDCPass> {
  using EmitSDCBase<EmitSDCPass>::EmitSDCBase;

  // Per-run flow configuration for clock barriers (parsed from options).
  SmallVector<GateRule> gateRules;
  SmallVector<std::string> sourceSubstrs;

  void runOnOperation() override {
    if (topModuleName.getValue().empty()) {
      getOperation()->emitError("synth-emit-sdc requires 'top-module-name'");
      return signalPassFailure();
    }

    ModuleOp top = getOperation();
    auto topSym =
        top.lookupSymbol(StringAttr::get(top.getContext(), topModuleName));
    auto topMod = dyn_cast_or_null<hw::HWModuleOp>(topSym);
    if (!topMod) {
      getOperation()->emitError("top module '" + topModuleName.getValue() +
                                "' not found or not an hw.module");
      return signalPassFailure();
    }

    auto &instanceGraph = getAnalysis<igraph::InstanceGraph>();

    // Parse clock-barrier flow configuration.
    gateRules.clear();
    sourceSubstrs.clear();
    auto splitComma = [](StringRef s) {
      SmallVector<StringRef> parts;
      while (!s.empty()) {
        auto [head, tail] = s.split(',');
        if (!head.trim().empty())
          parts.push_back(head.trim());
        s = tail;
      }
      return parts;
    };
    for (StringRef spec : splitComma(clockGateCells.getValue())) {
      auto eq = spec.split('=');
      auto cellOut = eq.first.split(':');
      if (eq.second.empty() || cellOut.second.empty())
        continue;
      gateRules.push_back(
          {cellOut.first.str(), cellOut.second.str(), eq.second.str()});
    }
    for (StringRef sub : splitComma(clockSourceCells.getValue()))
      sourceSubstrs.push_back(sub.str());

    // Modules reachable from the top (design hierarchy only).
    DenseSet<Operation *> reachable;
    SmallVector<igraph::ModuleOpInterface> worklist;
    if (auto *topNode = instanceGraph.lookup(topMod.getModuleNameAttr()))
      worklist.push_back(topNode->getModule());
    while (!worklist.empty()) {
      auto mod = worklist.pop_back_val();
      if (!reachable.insert(mod.getOperation()).second)
        continue;
      if (auto *node = instanceGraph.lookup(mod.getModuleNameAttr()))
        for (auto *child : *node)
          worklist.push_back(child->getTarget()->getModule());
    }

    // Resolve every flop clock in the reachable hierarchy. Resolutions
    // are memoized per (module, net): fanout-heavy clock nets resolve once.
    llvm::StringMap<unsigned> rootFlops;
    llvm::StringMap<unsigned> rootSharedFlops;
    llvm::StringMap<unsigned> sourceFlops;
    SmallVector<std::pair<std::string, std::string>> unconstrained;
    DenseMap<std::pair<Operation *, Value>, ClockResolution> memo;
    for (Operation *modOp : reachable) {
      auto hwMod = dyn_cast<hw::HWModuleOp>(modOp);
      if (!hwMod || isa<hw::HWModuleExternOp>(modOp))
        continue;
      for (auto &block : hwMod->getRegion(0)) {
        for (auto &op : block) {
          auto inst = dyn_cast<hw::InstanceOp>(op);
          if (!inst)
            continue;
          for (unsigned clockIdx : getClockOperandIndices(inst, top)) {
            Value clock = inst->getOperand(clockIdx);
            DenseSet<std::pair<Operation *, Value>> visited;
            ClockResolution resolution = resolveClock(
                clock, hwMod, topMod, instanceGraph, visited, memo);
            if (resolution.roots.empty()) {
              ++numUnconstrained;
              if (resolution.reason.empty())
                resolution.reason = "unresolved";
              if (!resolution.sources.empty()) {
                for (const auto &source : resolution.sources)
                  ++sourceFlops[source];
              } else {
                unconstrained.emplace_back(hwMod.getModuleName().str() + "/" +
                                               inst.getInstanceName().str(),
                                           resolution.reason);
              }
            } else if (resolution.sharedContext) {
              for (const auto &root : resolution.roots)
                ++rootSharedFlops[root];
            } else {
              for (const auto &root : resolution.roots)
                ++rootFlops[root];
            }
          }
        }
      }
    }

    // Emit.
    std::string error;
    auto file = mlir::openOutputFile(outputFile.getValue(), &error);
    if (!file) {
      getOperation()->emitError(error);
      return signalPassFailure();
    }
    auto &os = file->os();
    os << "# Generated by circt-synth synth-emit-sdc for top '"
       << topMod.getModuleName() << "'\n";
    SmallVector<StringRef> roots;
    for (const auto &entry : rootFlops)
      roots.push_back(entry.getKey());
    llvm::sort(roots);
    SmallVector<StringRef> sharedRoots;
    for (const auto &entry : rootSharedFlops)
      if (!rootFlops.contains(entry.getKey()))
        sharedRoots.push_back(entry.getKey());
    llvm::sort(sharedRoots);
    for (StringRef root : roots) {
      os << "# " << rootFlops[root] << " flops\n";
      os << "create_clock -name " << root << " -period "
         << llvm::format("%.3f", clockPeriodNs.getValue()) << " [get_ports {"
         << root << "}]\n";
      ++numClocks;
    }
    if (!roots.empty()) {
      std::string all;
      for (StringRef root : roots) {
        if (!all.empty())
          all += " ";
        all += root.str();
      }
      // Explicit port lists (no collection commands): portable across
      // OpenSTA builds. The first clock is the delay reference.
      // Shared-candidate ports are excluded from data inputs: they may be
      // clocks in some instantiation context.
      DenseSet<StringRef> clockPorts(roots.begin(), roots.end());
      clockPorts.insert(sharedRoots.begin(), sharedRoots.end());
      std::string dataInputs, dataOutputs;
      for (const auto &port : topMod.getPortList()) {
        if (port.isInput() && !clockPorts.contains(port.name.getValue())) {
          if (!dataInputs.empty())
            dataInputs += " ";
          dataInputs += port.name.getValue().str();
        }
        if (port.isOutput()) {
          if (!dataOutputs.empty())
            dataOutputs += " ";
          dataOutputs += port.name.getValue().str();
        }
      }
      if (!dataInputs.empty())
        os << "set_input_delay "
           << llvm::format("%.3f", inputDelayNs.getValue())
           << " -clock [get_clocks {" << roots.front() << "}] [get_ports {"
           << dataInputs << "}]\n";
      if (!dataOutputs.empty())
        os << "set_output_delay "
           << llvm::format("%.3f", outputDelayNs.getValue())
           << " -clock [get_clocks {" << roots.front() << "}] [get_ports {"
           << dataOutputs << "}]\n";
    }
    if (!sharedRoots.empty()) {
      os << "# Shared-context clock candidates:\n";
      for (StringRef root : sharedRoots)
        os << "#   " << root << " (" << rootSharedFlops[root] << " flops)"
           << "\n";
    }
    for (const auto &[loc, reason] : unconstrained)
      os << "# Unconstrained clock at " << loc << " (" << reason << ")\n";
    if (!sourceFlops.empty()) {
      SmallVector<StringRef> sources;
      for (const auto &entry : sourceFlops)
        sources.push_back(entry.getKey());
      llvm::sort(sources);
      os << "# On-chip clock sources (constrain manually, e.g. "
            "create_clock on the source pin):\n";
      for (StringRef source : sources)
        os << "#   " << source << " (" << sourceFlops[source] << " flops)"
           << "\n";
    }
    file->keep();
    markAllAnalysesPreserved();
  }

  /// Operand indices of clock pins (liberty `clock = "true"` inputs).
  SmallVector<unsigned> getClockOperandIndices(hw::InstanceOp inst,
                                               ModuleOp top) {
    SmallVector<unsigned> indices;
    auto *targetOp = top.lookupSymbol(inst.getModuleName());
    if (!targetOp)
      return indices;
    auto checkPorts = [&](ArrayRef<hw::PortInfo> ports) {
      unsigned operandIdx = 0;
      for (const auto &port : ports) {
        if (!port.isInput())
          continue;
        bool isClock = false;
        if (port.attrs) {
          if (auto pin = dyn_cast_or_null<DictionaryAttr>(
                  port.attrs.get("synth.liberty.pin")))
            if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
              isClock = clock.getValue() == "true";
        }
        if (isClock && operandIdx < inst.getNumOperands())
          indices.push_back(operandIdx);
        ++operandIdx;
      }
    };
    if (auto hwMod = dyn_cast<hw::HWModuleOp>(targetOp))
      checkPorts(hwMod.getPortList());
    else if (auto ext = dyn_cast<hw::HWModuleExternOp>(targetOp))
      checkPorts(ext.getPortList());
    return indices;
  }

  /// Output port name driven by an instance result, null when unknown.
  StringAttr instanceOutputPortName(hw::InstanceOp inst, Value value) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return {};
    auto *targetOp = getOperation().lookupSymbol(inst.getModuleName());
    if (!targetOp)
      return {};
    SmallVector<hw::PortInfo> ports;
    if (auto hwMod = dyn_cast<hw::HWModuleOp>(targetOp))
      ports = hwMod.getPortList();
    else if (auto ext = dyn_cast<hw::HWModuleExternOp>(targetOp))
      ports = ext.getPortList();
    else
      return {};
    unsigned outputIdx = 0;
    for (const auto &port : ports) {
      if (!port.isOutput())
        continue;
      if (outputIdx == result.getResultNumber())
        return port.name;
      ++outputIdx;
    }
    return {};
  }

  /// True when the value is driven by a configured on-chip clock source.
  bool isClockSource(hw::InstanceOp inst, Value value) {
    if (sourceSubstrs.empty())
      return false;
    if (!instanceOutputPortName(inst, value))
      return false;
    StringRef target = inst.getModuleName();
    return llvm::any_of(sourceSubstrs, [&](const std::string &sub) {
      return target.contains(sub);
    });
  }

  /// Human-readable driving port for source reports.
  std::string clockPortName(hw::InstanceOp inst, Value value) {
    if (auto name = instanceOutputPortName(inst, value))
      return name.getValue().str();
    return "?";
  }

  /// Apply a configured clock-gate rule (Cell:out=in): resolve through the
  /// named input. Returns true when a rule matched (resolution merged).
  bool
  applyGateRule(hw::InstanceOp inst, Value value,
                igraph::ModuleOpInterface curMod, hw::HWModuleOp topMod,
                igraph::InstanceGraph &graph,
                DenseSet<std::pair<Operation *, Value>> &visited,
                DenseMap<std::pair<Operation *, Value>, ClockResolution> &memo,
                ClockResolution &resolution) {
    if (gateRules.empty())
      return false;
    auto outName = instanceOutputPortName(inst, value);
    if (!outName)
      return false;
    for (const auto &rule : gateRules) {
      if (inst.getModuleName() != rule.cell || outName.getValue() != rule.out)
        continue;
      auto *targetOp = getOperation().lookupSymbol(inst.getModuleName());
      if (!targetOp)
        return false;
      SmallVector<hw::PortInfo> ports;
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(targetOp))
        ports = hwMod.getPortList();
      else if (auto ext = dyn_cast<hw::HWModuleExternOp>(targetOp))
        ports = ext.getPortList();
      unsigned operandIdx = 0;
      for (const auto &port : ports) {
        if (port.isOutput())
          continue;
        if (port.isInput() && port.name.getValue() == rule.in &&
            operandIdx < inst.getNumOperands()) {
          ClockResolution sub =
              resolveClock(inst.getOperand(operandIdx), curMod, topMod, graph,
                           visited, memo);
          mergeSubResolution(sub, resolution);
          return true;
        }
      }
      return false;
    }
    return false;
  }

  /// Descend into a hierarchical target: resolve the driver of the output
  /// port feeding `value`. Exact (follows the real driver net); a flop-Q
  /// driver correctly reports as generated further down the trace.
  /// Returns true when descended (resolution merged, even if empty).
  bool descendIntoTarget(
      hw::InstanceOp inst, Value value, hw::HWModuleOp topMod,
      igraph::InstanceGraph &graph,
      DenseSet<std::pair<Operation *, Value>> &visited,
      DenseMap<std::pair<Operation *, Value>, ClockResolution> &memo,
      ClockResolution &resolution) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    auto *targetOp = getOperation().lookupSymbol(inst.getModuleName());
    auto targetMod = dyn_cast_or_null<hw::HWModuleOp>(targetOp);
    if (!targetMod)
      return false;
    auto *body = targetMod.getBodyBlock();
    if (!body)
      return false;
    auto output = dyn_cast_or_null<hw::OutputOp>(body->getTerminator());
    if (!output || result.getResultNumber() >= output->getNumOperands())
      return false;
    auto targetIF = dyn_cast<igraph::ModuleOpInterface>(targetOp);
    if (!targetIF)
      return false;
    ClockResolution sub =
        resolveClock(output->getOperand(result.getResultNumber()), targetIF,
                     topMod, graph, visited, memo);
    mergeSubResolution(sub, resolution);
    return true;
  }

  /// Merge a sub-resolution (roots, sources, sharing, reason).
  static void mergeSubResolution(const ClockResolution &sub,
                                 ClockResolution &resolution) {
    resolution.roots.append(sub.roots.begin(), sub.roots.end());
    resolution.sources.append(sub.sources.begin(), sub.sources.end());
    resolution.sharedContext = resolution.sharedContext || sub.sharedContext;
    if (!sub.reason.empty() && resolution.reason.empty())
      resolution.reason = sub.reason;
  }

  /// Trace a clock net to top-level input ports. Results memoized per
  ClockResolution
  resolveClock(Value value, igraph::ModuleOpInterface curMod,
               hw::HWModuleOp topMod, igraph::InstanceGraph &graph,
               DenseSet<std::pair<Operation *, Value>> &visited,
               DenseMap<std::pair<Operation *, Value>, ClockResolution> &memo) {
    auto memoKey = std::make_pair(curMod.getOperation(), value);
    auto memoIt = memo.find(memoKey);
    if (memoIt != memo.end())
      return memoIt->second;
    ClockResolution resolution;
    resolveClockImpl(value, curMod, topMod, graph, visited, memo, resolution);
    memo[memoKey] = resolution;
    return resolution;
  }

  void resolveClockImpl(
      Value value, igraph::ModuleOpInterface curMod, hw::HWModuleOp topMod,
      igraph::InstanceGraph &graph,
      DenseSet<std::pair<Operation *, Value>> &visited,
      DenseMap<std::pair<Operation *, Value>, ClockResolution> &memo,
      ClockResolution &resolution) {
    if (!visited.insert({curMod.getOperation(), value}).second) {
      resolution.reason = "clock loop";
      return;
    }
    // Through single-input cells (buffers, inverters).
    if (auto *defOp = value.getDefiningOp()) {
      if (auto inst = dyn_cast<hw::InstanceOp>(defOp)) {
        // On-chip clock source: record for manual constraint, stop.
        if (isClockSource(inst, value)) {
          resolution.sources.push_back(inst.getModuleName().str() + "/" +
                                       clockPortName(inst, value));
          resolution.reason = "on-chip clock source";
          return;
        }
        // Clock gate: attribute to the gate input (standard first-order
        // SDC; gating checks out of scope).
        if (applyGateRule(inst, value, curMod, topMod, graph, visited, memo,
                          resolution))
          return;
        // Hierarchical target: descend to the real driver of the output
        // port. Exact (follows the driver net); flop-Q-driven outputs
        // correctly report as generated further down.
        if (descendIntoTarget(inst, value, topMod, graph, visited, memo,
                              resolution))
          return;
        // Leaf single-input cell (or unknown target): follow the input.
        if (inst.getNumOperands() == 1) {
          resolution = resolveClock(inst.getOperand(0), curMod, topMod, graph,
                                    visited, memo);
          return;
        }
        resolution.reason =
            "driven by multi-input cell '" + inst.getModuleName().str() + "'";
        return;
      }
      if (isa<hw::ConstantOp>(value.getDefiningOp())) {
        resolution.reason = "tied to constant";
        return;
      }
      resolution.reason = "driven by combinational logic";
      return;
    }
    // Block argument: top port or ascend through instantiations.
    auto blockArg = cast<BlockArgument>(value);
    if (curMod.getOperation() == topMod.getOperation()) {
      unsigned argNum = blockArg.getArgNumber();
      unsigned inputIdx = 0;
      for (const auto &port : topMod.getPortList()) {
        if (!port.isInput())
          continue;
        if (inputIdx == argNum) {
          resolution.roots.push_back(port.name.getValue().str());
          return;
        }
        ++inputIdx;
      }
      resolution.reason = "top argument is not an input port";
      return;
    }
    auto *node =
        graph.lookup(cast<igraph::ModuleOpInterface>(curMod.getOperation())
                         .getModuleNameAttr());
    if (!node) {
      resolution.reason = "module not in instance graph";
      return;
    }
    bool ascended = false;
    if (!node->hasOneUse())
      resolution.sharedContext = true;
    for (auto *record : node->uses()) {
      Operation *instanceOp = record->getInstance().getOperation();
      unsigned argNum = blockArg.getArgNumber();
      if (argNum >= instanceOp->getNumOperands())
        continue;
      ascended = true;
      ClockResolution parent = resolveClock(instanceOp->getOperand(argNum),
                                            record->getParent()->getModule(),
                                            topMod, graph, visited, memo);
      mergeSubResolution(parent, resolution);
    }
    if (!ascended)
      resolution.reason = "uninstantiated module input";
  }
};

} // namespace
