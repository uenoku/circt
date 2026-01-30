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
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace circt::synth::timing;

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
  if (isa<aig::AndInverterOp>(op))
    return {1, 0.0};

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
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<DelayModel> circt::synth::timing::createDefaultDelayModel() {
  return std::make_unique<AIGLevelDelayModel>();
}
