//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_H
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_H

#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/IR/Value.h"
#include <mlir/IR/Value.h>

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.h.inc"

namespace circt {
namespace synth {
struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Helper struct to represent ordered children of a majority operation
/// Uses PointerIntPair to store both the Value and its inversion flag
struct InvertibleOperand {
  llvm::PointerIntPair<Value, 1, bool> value;
  InvertibleOperand(Value value, bool inverted) : value({value, inverted}) {}
  explicit InvertibleOperand(Value value) : value({value, false}) {}

  bool isInverted() const { return value.getInt(); }
  Value getValue() const { return value.getPointer(); }
};

struct TimedInvertibleOperand : InvertibleOperand {
  int64_t depth;
  TimedInvertibleOperand(Value value, bool inverted, int64_t depth)
      : InvertibleOperand(value, inverted), depth(depth) {}
  bool operator<(const TimedInvertibleOperand &other) const {
    return depth < other.depth;
  }
};

struct OrderedOperands {

  // PointerIntPair stores Value pointer and inversion flag (1 bit)
  SmallVector<TimedInvertibleOperand, 3> invertibleOperadns;

  OrderedOperands(OperandRange operands, ArrayRef<bool> inversions,
                  ArrayRef<int64_t> depths);

  static FailureOr<OrderedOperands>
  get(mig::MajorityInverterOp op, IncrementalLongestPathAnalysis *analysis);
  static FailureOr<OrderedOperands>
  get(aig::AndInverterOp op, IncrementalLongestPathAnalysis *analysis);

  void dump(llvm::raw_ostream &os) const {
    for (size_t i = 0; i < invertibleOperadns.size(); ++i) {
      os << "  Child " << i << ": " << invertibleOperadns[i].value.getPointer()
         << " (inverted: " << invertibleOperadns[i].value.getInt()
         << ", depth: " << invertibleOperadns[i].depth << ")\n";
    }
  }

  Value getValue(size_t idx) const {
    return invertibleOperadns[idx].getValue();
  }
  bool isInverted(size_t idx) const {
    return invertibleOperadns[idx].isInverted();
  }
  int64_t getDepth(size_t idx) const { return invertibleOperadns[idx].depth; }

  auto operator[](size_t idx) const { return invertibleOperadns[idx]; }
};
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_H
