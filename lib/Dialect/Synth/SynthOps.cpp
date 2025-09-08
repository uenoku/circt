//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/LogicalResult.h"

using namespace circt;
using namespace circt::synth::mig;

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.cpp.inc"

LogicalResult MajorityInverterOp::verify() {
  if (getNumOperands() % 2 != 1)
    return emitOpError("requires an odd number of operands");

  return success();
}

LogicalResult MajorityInverterOp::canonicalize(MajorityInverterOp op,
                                               PatternRewriter &rewriter) {
  if (op.getNumOperands() == 1) {
    if (op.getInverted()[0])
      return failure();
    rewriter.replaceOp(op, op.getOperand(0));
    return success();
  }

  // For now, only support 3 operands.
  if (op.getNumOperands() != 3)
    return failure();

  // Return if the idx-th operand is a constant (inverted if necessary),
  // otherwise return std::nullopt.
  auto getConstant = [&](unsigned index) -> std::optional<llvm::APInt> {
    APInt value;
    if (mlir::matchPattern(op.getInputs()[index], mlir::m_ConstantInt(&value)))
      return op.isInverted(index) ? ~value : value;
    return std::nullopt;
  };

  // Replace the op with the idx-th operand (inverted if necessary).
  auto replaceWithIndex = [&](int index) {
    bool inverted = op.isInverted(index);
    if (inverted)
      rewriter.replaceOpWithNewOp<MajorityInverterOp>(
          op, op.getType(), op.getOperand(index), true);
    else
      rewriter.replaceOp(op, op.getOperand(index));
    return success();
  };

  // Pattern match following cases:
  // maj_inv(x, x, y) -> x
  // maj_inv(x, y, not y) -> x
  for (int i = 0; i < 2; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      int k = 3 - (i + j);
      assert(k >= 0 && k < 3);
      // If we have two identical operands, we can fold.
      if (op.getOperand(i) == op.getOperand(j)) {
        // If they are inverted differently, we can fold to the third.
        if (op.isInverted(i) != op.isInverted(j)) {
          return replaceWithIndex(k);
        }
        rewriter.replaceOp(op, op.getOperand(i));
        return success();
      }

      // If i and j are constant.
      if (auto c1 = getConstant(i)) {
        if (auto c2 = getConstant(j)) {
          // If both constants are equal, we can fold.
          if (*c1 == *c2) {
            rewriter.replaceOpWithNewOp<hw::ConstantOp>(
                op, op.getType(), mlir::IntegerAttr::get(op.getType(), *c1));
            return success();
          }
          // If constants are complementary, we can fold.
          if (*c1 == ~*c2)
            return replaceWithIndex(k);
        }
      }
    }
  }
  return failure();
}
