//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SynthPreprocess pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SYNTHPREPROCESS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;
using namespace hw;
using namespace seq;

namespace {

/// Convert EICG_wrapper instance to seq.clock_gate
static void convertClockGateInstance(InstanceOp inst, OpBuilder &builder) {
  // Find the ports by name: in, en, test_en (optional)
  auto argNames = inst.getArgNames();

  Value inClock, enable, testEnable;

  // Match ports by name
  for (auto [idx, name] : llvm::enumerate(argNames)) {
    auto nameStr = cast<StringAttr>(name).getValue();
    if (nameStr == "in")
      inClock = inst.getOperand(idx);
    else if (nameStr == "en")
      enable = inst.getOperand(idx);
    else if (nameStr == "test_en")
      testEnable = inst.getOperand(idx);
  }

  // Verify we found required ports (in and en are required, test_en is optional)
  if (!inClock || !enable)
    return;

  builder.setInsertionPoint(inst);

  // Convert i1 clock to !seq.clock if needed
  if (!isa<seq::ClockType>(inClock.getType()))
    inClock = builder.create<seq::ToClockOp>(inst.getLoc(), inClock);

  // Create seq.clock_gate (dropping test_en for now as requested)
  auto clockGate = builder.create<seq::ClockGateOp>(inst.getLoc(), inClock, enable);

  // Convert back to i1 if the original output was i1
  Value result = clockGate.getResult();
  if (inst.getNumResults() > 0 &&
      !isa<seq::ClockType>(inst.getResult(0).getType()))
    result = builder.create<seq::FromClockOp>(inst.getLoc(), result);

  // Replace the instance
  inst.replaceAllUsesWith(ValueRange{result});
  inst.erase();
}

/// Convert SimpleClockMux_wrapper instance to seq.clock_mux
static void convertClockMuxInstance(InstanceOp inst, OpBuilder &builder) {
  // Find the ports by name: sel, clk0_in/clk0, clk1_in/clk1
  auto argNames = inst.getArgNames();

  Value sel, clk0, clk1;

  // Match ports by name
  for (auto [idx, name] : llvm::enumerate(argNames)) {
    auto nameStr = cast<StringAttr>(name).getValue();
    if (nameStr == "sel")
      sel = inst.getOperand(idx);
    else if (nameStr == "clk0" || nameStr == "clk0_in")
      clk0 = inst.getOperand(idx);
    else if (nameStr == "clk1" || nameStr == "clk1_in")
      clk1 = inst.getOperand(idx);
  }

  // Verify we found all required ports
  if (!sel || !clk0 || !clk1)
    return;

  builder.setInsertionPoint(inst);

  // Convert i1 clocks to !seq.clock if needed
  if (!isa<seq::ClockType>(clk0.getType()))
    clk0 = builder.create<seq::ToClockOp>(inst.getLoc(), clk0);
  if (!isa<seq::ClockType>(clk1.getType()))
    clk1 = builder.create<seq::ToClockOp>(inst.getLoc(), clk1);

  // Create seq.clock_mux: when sel is true, use clk1, otherwise clk0
  auto clockMux = builder.create<seq::ClockMuxOp>(inst.getLoc(), sel, clk1, clk0);

  // Convert back to i1 if the original output was i1
  Value result = clockMux.getResult();
  if (inst.getNumResults() > 0 &&
      !isa<seq::ClockType>(inst.getResult(0).getType()))
    result = builder.create<seq::FromClockOp>(inst.getLoc(), result);

  // Replace the instance
  inst.replaceAllUsesWith(ValueRange{result});
  inst.erase();
}

struct SynthPreprocessPass
    : public impl::SynthPreprocessBase<SynthPreprocessPass> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getContext());

    SmallVector<InstanceOp> clockMuxInstances;
    SmallVector<InstanceOp> clockGateInstances;
    SmallVector<seq::FirRegOp> regsToConvert;

    // Collect instances of SimpleClockMux_wrapper and EICG_wrapper
    module.walk([&](InstanceOp inst) {
      if (inst.getModuleName() == "SimpleClockMux_wrapper")
        clockMuxInstances.push_back(inst);
      else if (inst.getModuleName() == "EICG_wrapper")
        clockGateInstances.push_back(inst);
    });

    // Collect clock divider patterns
    module.walk([&](seq::FirRegOp reg) {
      // Check if this is an i1 register with async reset
      if (!reg.getType().isInteger(1) || !reg.getIsAsync())
        return;

      // Check if reset value is constant true
      auto resetValue = reg.getResetValue();
      if (!resetValue)
        return;

      auto constOp = resetValue.getDefiningOp<hw::ConstantOp>();
      if (!constOp || !constOp.getValue().isOne())
        return;

      // Check if the input is synth.aig.and_inv not %clock_div
      auto andInv = reg.getNext().getDefiningOp<aig::AndInverterOp>();
      if (!andInv || andInv.getNumOperands() != 1)
        return;

      if (!andInv.isInverted(0) || andInv.getOperand(0) != reg.getResult())
        return;

      // This is a clock divider pattern
      regsToConvert.push_back(reg);
    });

    // Convert SimpleClockMux_wrapper instances
    for (auto inst : clockMuxInstances)
      convertClockMuxInstance(inst, builder);

    // Convert EICG_wrapper instances
    for (auto inst : clockGateInstances)
      convertClockGateInstance(inst, builder);

    // Convert clock divider patterns
    for (auto reg : regsToConvert) {
      auto clock = reg.getClk();
      auto andInv = reg.getNext().getDefiningOp<aig::AndInverterOp>();

      builder.setInsertionPoint(reg);

      // Create clock_div by 1 (divide by 2)
      auto clockDiv = builder.create<seq::ClockDividerOp>(reg.getLoc(), clock, 1);

      // Convert to i1 for compatibility
      auto fromClock = builder.create<seq::FromClockOp>(reg.getLoc(), clockDiv);

      // Replace the register
      reg.replaceAllUsesWith(fromClock.getResult());
      reg.erase();

      // Erase the and_inv operation if it has no other uses
      if (andInv && andInv.use_empty())
        andInv.erase();
    }
  }
};

} // namespace

