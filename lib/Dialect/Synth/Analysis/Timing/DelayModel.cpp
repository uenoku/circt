//===- DelayModel.cpp - Pluggable Delay Model Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace circt;
using namespace circt::synth::timing;

static int64_t getAttrDelay(Operation *op, StringRef attrName) {
  if (!op)
    return -1;
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return -1;
}

static std::optional<unsigned> getOperandIndex(Operation *op, Value value) {
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (operand == value)
      return idx;
  }
  return std::nullopt;
}

static std::optional<unsigned> getResultIndex(Operation *op, Value value) {
  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (result == value)
      return idx;
  }
  return std::nullopt;
}

static int64_t getPerArcDelay(Operation *op, Value inputValue,
                              Value outputValue) {
  if (!op)
    return -1;

  auto dict = op->getAttrOfType<DictionaryAttr>("synth.liberty.arc_delay_ps");
  if (!dict)
    return -1;

  auto inIdx = getOperandIndex(op, inputValue);
  auto outIdx = getResultIndex(op, outputValue);
  if (!inIdx || !outIdx)
    return -1;

  llvm::SmallString<32> key;
  key += "i";
  key += llvm::utostr(*inIdx);
  key += "_o";
  key += llvm::utostr(*outIdx);

  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(key)))
    return attr.getInt();
  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get("default")))
    return attr.getInt();
  return -1;
}

//===----------------------------------------------------------------------===//
// UnitDelayModel
//===----------------------------------------------------------------------===//

DelayResult UnitDelayModel::computeDelay(const DelayContext &ctx) const {
  auto *op = ctx.op;
  // Zero-cost wiring operations
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op))
    return {0, 0.0};
  // All other ops get unit delay
  return {1, 0.0};
}

//===----------------------------------------------------------------------===//
// AIGLevelDelayModel
//===----------------------------------------------------------------------===//

DelayResult AIGLevelDelayModel::computeDelay(const DelayContext &ctx) const {
  auto *op = ctx.op;

  // AIG operations
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op))
    return {llvm::Log2_64_Ceil(andOp.getNumOperands()), 0.0};

  // Comb operations
  if (isa<comb::MuxOp>(op))
    return {1, 0.0};
  if (auto andOp = dyn_cast<comb::AndOp>(op))
    return {llvm::Log2_64_Ceil(andOp.getNumOperands()), 0.0};
  if (auto orOp = dyn_cast<comb::OrOp>(op))
    return {llvm::Log2_64_Ceil(orOp.getNumOperands()), 0.0};
  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return {llvm::Log2_64_Ceil(xorOp.getNumOperands()), 0.0};

  // Zero-cost operations (bit manipulation)
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op))
    return {0, 0.0};

  // Default cost
  return {1, 0.0};
}

//===----------------------------------------------------------------------===//
// NLDMDelayModel
//===----------------------------------------------------------------------===//

DelayResult NLDMDelayModel::computeDelay(const DelayContext &ctx) const {
  if (int64_t delay = getPerArcDelay(ctx.op, ctx.inputValue, ctx.outputValue);
      delay >= 0)
    return {delay, 0.0};

  if (int64_t delay = getAttrDelay(ctx.op, "synth.liberty.delay_ps");
      delay >= 0)
    return {delay, 0.0};

  return fallback.computeDelay(ctx);
}

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<DelayModel> circt::synth::timing::createDefaultDelayModel() {
  return std::make_unique<AIGLevelDelayModel>();
}

std::unique_ptr<DelayModel> circt::synth::timing::createNLDMDelayModel() {
  return std::make_unique<NLDMDelayModel>();
}
