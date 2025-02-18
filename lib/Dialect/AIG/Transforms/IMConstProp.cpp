//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers multi-bit AIG operations to single-bit ones.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aig-imconstprop"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_IMCONSTPROP
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

class LatticeValue {
  // Each bit represents 4-state lattice value.
  //    unknown (0, 1)
  //   /           \
  //  1 (1, 0)      0 (0, 0)
  //   \            /
  //    overdefined (1, 1)
  APInt value, unknown;

public:
  /// Initialize a lattice value with "Unknown".
  /*implicit*/ LatticeValue(size_t numBits)
      : value(numBits, 0), unknown(APInt::getAllOnes(numBits)) {}

  /// Compute a mask of all the 0 bits in this integer.
  APInt getZeroBits() const { return ~value & ~unknown; }

  void markOverdefined() {
    value.setAllBits();
    unknown.setAllBits();
  }

  /// Compute the logical AND of this integer and another. This implements the
  /// following bit-wise truth table:
  ///     0 1 t b
  ///   +--------
  /// 0 | 0 0 0 0
  /// 1 | 0 1 t b
  /// t | 0 t t t
  /// b | 0 b t b
  LatticeValue &operator&=(const LatticeValue &other) {
    auto zeros = getZeroBits() | other.getZeroBits();
    value &= other.value;
    unknown |= other.unknown;
    unknown &= ~zeros;
    return *this;
  }
  LatticeValue operator~() const {
    auto v = *this;
    v.value = ~v.value;
    return v;
  }

  static LatticeValue getOverdefined(size_t numBits) {
    LatticeValue result(numBits);
    result.value.setAllBits();
    result.unknown.setAllBits();
    return result;
  }

  static LatticeValue getUnknown(size_t numBits) {
    LatticeValue result(numBits);
    result.value.setAllBits();
    result.unknown.setAllBits();
    return result;
  }

  void setRange(size_t start, const LatticeValue &value) {
    // this->value.setBits(start, start + value.value.getBitWidth(), value.value);
    // this->unknown.setBits(start, start + value.unknown.getBitWidth(),
    //                       value.unknown);

    unknown.clearAllBits();
  }

  static LatticeValue getConstant(size_t numBits) {
    LatticeValue result(numBits);
    result.value.setAllBits();
    result.unknown.setAllBits();
    return result;
  }

  bool isAllOverdefined() const {
    return value.isAllOnes() && unknown.isAllOnes();
  }

  bool operator==(const LatticeValue &other) const {
    return other.value == value && other.unknown == unknown;
  }
};

namespace {
struct IMConstPropPass : public impl::IMConstPropBase<IMConstPropPass> {
  void runOnOperation() override;
  SmallVector<Value, 64> changedValueWorklist;
  DenseMap<Value, LatticeValue> latticeValues;
  LatticeValue getLatticeValue(Value value) { return latticeValues.at(value); }
  void visitOperation(Operation *op);
  void markOverdefined(Value value);
  void setLatticeValue(Value value, LatticeValue latticeValue);
  void visitConcat(comb::ConcatOp op);
  void visitReplicate(comb::ReplicateOp op);
  void visitExtract(comb::ExtractOp op);
  void visitConstant(comb::ExtractOp op);
};
} // namespace

void IMConstPropPass::setLatticeValue(Value value, LatticeValue latticeValue) {
  if (latticeValues.at(value) == latticeValue)
    return;

  latticeValues[value] = std::move(latticeValue);
  changedValueWorklist.push_back(value);
}

void IMConstPropPass::runOnOperation() {}

void IMConstPropPass::visitOperation(Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<aig::AndInverterOp>([&](auto op) {
        LatticeValue result = getLatticeValue(op.getInputs().front());
        for (auto input : op.getInputs().drop_front()) {
          result &= latticeValues.at(input);
        }
        setLatticeValue(op, result);
      })
      .Case<comb::ConcatOp>([&](comb::ConcatOp op) {
        LatticeValue result(op.getResult().getType().getIntOrFloatBitWidth());
        for (auto input : op.getInputs().drop_front()) {
          auto &value = latticeValues.at(input);
        }
      })
      .Default([&](auto _) {
        for (auto result : op->getResults())
          markOverdefined(result);
      });
}
