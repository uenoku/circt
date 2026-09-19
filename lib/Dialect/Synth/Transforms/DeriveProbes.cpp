//===- DeriveProbes.cpp - Derive timing probes from clocks ---------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Places one `synth.timing_probe` on every flop D input whose clock traces
// to a listed clock port, with required_time = period. Clock tracing walks
// nets upward through single-input cells and EICG clock gates using liberty
// clock-pin metadata; anything else (multi-input gating, generated clocks)
// conservatively yields no probes. Probes are consumed and erased by
// `synth-buffer-high-fanout`. All ordering is file/IR order.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
namespace synth {
#define GEN_PASS_DECL_DERIVEPROBES
#define GEN_PASS_DEF_DERIVEPROBES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-derive-probes"

namespace {

static bool isLibModule(hw::HWModuleOp module) {
  return module->hasAttr("synth.liberty.cell") ||
         module->hasAttr("synth.dont_use") ||
         module->hasAttr("synth.mapping_cost") ||
         module->hasAttr("synth.supergate_candidate") ||
         module.getModuleName().starts_with("__supergate_");
}

static void splitPorts(hw::HWModuleOp module, SmallVector<hw::PortInfo> &inputs,
                       SmallVector<hw::PortInfo> &outputs) {
  for (const auto &port : module.getPortList()) {
    if (port.isInput())
      inputs.push_back(port);
    else if (port.isOutput())
      outputs.push_back(port);
  }
}

// Body contains a clock gate (EICG-style wrapper).
static bool hasClockGate(hw::HWModuleOp module) {
  if (!module.getBodyBlock())
    return false;
  return module.getBodyBlock()
      ->walk([](Operation *op) {
        if (isa<ClockGateOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Body contains an edge-triggered flop (latches excluded: borrowing
// semantics need required-time treatment probes don't provide).
static bool hasFlop(hw::HWModuleOp module) {
  if (!module.getBodyBlock())
    return false;
  return module.getBodyBlock()
      ->walk([](Operation *op) {
        if (isa<FFOp>(op))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

// Input port index with the liberty clock pin (if any).
static int clockPinIndex(hw::HWModuleOp module) {
  SmallVector<hw::PortInfo> inputs, outputs;
  splitPorts(module, inputs, outputs);
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (inputs[i].attrs) {
      if (auto pin = dyn_cast_or_null<DictionaryAttr>(
              inputs[i].attrs.get("synth.liberty.pin")))
        if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
          if (clock.getValue() == "true")
            return (int)i;
    }
  }
  return -1;
}

struct DeriveProbesPass
    : public circt::synth::impl::DeriveProbesBase<DeriveProbesPass> {
  using DeriveProbesBase<DeriveProbesPass>::DeriveProbesBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();
    DenseSet<StringRef> clocks;
    {
      SmallVector<StringRef> parts;
      StringRef(probeClocks).split(parts, ',');
      for (auto part : parts) {
        part = part.trim();
        if (!part.empty())
          clocks.insert(part);
      }
    }
    if (clocks.empty())
      return markAllAnalysesPreserved();
    int64_t required = llround(probePeriodNs * 1000.0);
    if (required <= 0) {
      top->emitError("probe-period-ns must be positive");
      signalPassFailure();
      return;
    }
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module))
        continue;
      for (auto &block : module->getRegion(0))
        for (auto &op : block)
          if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
            auto *target = top.lookupSymbol(inst.getModuleName());
            auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
            if (targetModule && !isLibModule(targetModule))
              designParents[target].push_back({module, inst});
          }
    }

    bool changed = false;
    for (auto module : top.getOps<hw::HWModuleOp>()) {
      if (isLibModule(module) || !module.getBodyBlock())
        continue;
      changed |= deriveModule(module, clocks, required);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  // Memoized "net is clocked" per value. Upward walk with cycle guard.
  DenseMap<Value, char> clockedMemo;
  DenseSet<Value> clockedActive;
  // Design parent instances per module (built once, file order).
  DenseMap<Operation *, SmallVector<std::pair<hw::HWModuleOp, hw::InstanceOp>>>
      designParents;

  bool isClocked(ModuleOp top, const DenseSet<StringRef> &clocks,
                 hw::HWModuleOp module, Value value) {
    auto memo = clockedMemo.find(value);
    if (memo != clockedMemo.end())
      return memo->second;
    if (!clockedActive.insert(value).second)
      return false;
    bool result = computeClocked(top, clocks, module, value);
    clockedActive.erase(value);
    clockedMemo[value] = result;
    return result;
  }

  bool computeClocked(ModuleOp top, const DenseSet<StringRef> &clocks,
                      hw::HWModuleOp module, Value value) {
    auto parentIt = designParents.find(module.getOperation());
    bool hasDesignParent =
        parentIt != designParents.end() && !parentIt->second.empty();
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      unsigned argIdx = blockArg.getArgNumber();
      // Block args align with input ports in order.
      SmallVector<hw::PortInfo> inputs, outputs;
      splitPorts(module, inputs, outputs);
      if (!hasDesignParent) {
        // Root: clocked iff the port is listed.
        return argIdx < inputs.size() &&
               clocks.contains(inputs[argIdx].name.getValue());
      }
      // Otherwise clocked iff any parent driver is clocked (permissive:
      // over-probing only adds buffers, never breaks clocks).
      for (auto [parent, inst] : parentIt->second) {
        if (argIdx >= inst.getNumOperands())
          continue;
        if (isClocked(top, clocks, parent, inst.getOperand(argIdx)))
          return true;
      }
      return false;
    }
    auto *defOp = value.getDefiningOp();
    if (!defOp)
      return false;
    if (auto inst = dyn_cast<hw::InstanceOp>(defOp)) {
      // Single-input cells (buffers, inverters) preserve clockedness.
      if (inst.getNumOperands() == 1)
        return isClocked(top, clocks, module, inst.getOperand(0));
      // EICG-style clock gates: follow the clock pin.
      auto *target = top.lookupSymbol(inst.getModuleName());
      auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
      if (targetModule && hasClockGate(targetModule)) {
        int clkIdx = clockPinIndex(targetModule);
        if (clkIdx >= 0 && (unsigned)clkIdx < inst.getNumOperands())
          return isClocked(top, clocks, module,
                           inst.getOperand((unsigned)clkIdx));
      }
      return false;
    }
    return false;
  }

  bool deriveModule(hw::HWModuleOp module, const DenseSet<StringRef> &clocks,
                    int64_t required) {
    bool changed = false;
    OpBuilder builder(module.getContext());
    // Collect first; insertion mutates use lists.
    SmallVector<hw::InstanceOp> flops;
    for (auto &block : module->getRegion(0))
      for (auto &op : block) {
        auto inst = dyn_cast<hw::InstanceOp>(op);
        if (!inst)
          continue;
        auto *target = getOperation().lookupSymbol(inst.getModuleName());
        auto targetModule = dyn_cast_or_null<hw::HWModuleOp>(target);
        if (!targetModule || !hasFlop(targetModule))
          continue;
        flops.push_back(inst);
      }
    for (auto inst : flops) {
      auto *target = getOperation().lookupSymbol(inst.getModuleName());
      auto targetModule = cast<hw::HWModuleOp>(target);
      SmallVector<hw::PortInfo> inputs, outputs;
      splitPorts(targetModule, inputs, outputs);
      int dataIdx = -1, clkIdx = -1;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        if (inputs[i].name.getValue() == "D" && inputs[i].type.isInteger(1))
          dataIdx = (int)i;
        if (inputs[i].attrs) {
          if (auto pin = dyn_cast_or_null<DictionaryAttr>(
                  inputs[i].attrs.get("synth.liberty.pin")))
            if (auto clock = dyn_cast_or_null<StringAttr>(pin.get("clock")))
              if (clock.getValue() == "true")
                clkIdx = (int)i;
        }
      }
      if (dataIdx < 0 || clkIdx < 0 ||
          (unsigned)dataIdx >= inst.getNumOperands() ||
          (unsigned)clkIdx >= inst.getNumOperands())
        continue;
      if (!isClocked(getOperation(), clocks, module,
                     inst.getOperand((unsigned)clkIdx)))
        continue;
      builder.setInsertionPoint(inst);
      auto probe = synth::TimingProbeOp::create(
          builder, inst.getLoc(), inst.getOperand((unsigned)dataIdx), required);
      inst.setOperand((unsigned)dataIdx, probe.getResult());
      ++numProbesDerived;
      changed = true;
    }
    return changed;
  }
};

} // namespace
