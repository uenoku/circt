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
#include "circt/Support/Namespace.h"
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

namespace circt {
namespace synth {
namespace aig {
class AndInverterOp;
} // namespace aig
namespace mig {
class MajorityInverterOp;
} // namespace mig

struct AndInverterVariadicOpConversion
    : mlir::OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(aig::AndInverterOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// This function performs a topological sort on the operations within each
/// block of graph regions in the given operation. It uses MLIR's topological
/// sort utility as a wrapper, ensuring that operations are ordered such that
/// all operands are defined before their uses. The `isOperandReady` callback
/// allows customization of when an operand is considered ready for sorting.
LogicalResult topologicallySortGraphRegionBlocks(
    mlir::Operation *op,
    llvm::function_ref<bool(mlir::Value, mlir::Operation *)> isOperandReady);

/// Helper struct to represent a value that may be inverted.
struct InvertibleValue {
  llvm::PointerIntPair<Value, 1, bool> value;
  InvertibleValue(Value value, bool inverted = false)
      : value({value, inverted}) {}

  void operator^=(bool invert) { value.setInt(value.getInt() ^ invert); }
  InvertibleValue operator^(bool invert) const {
    return InvertibleValue(value.getPointer(), value.getInt() ^ invert);
  }
  bool operator==(const InvertibleValue &other) const {
    return value == other.value;
  }
  InvertibleValue operator!() const {
    return InvertibleValue(value.getPointer(), !value.getInt());
  }

  /// Returns true if the value is the same as the other value and inverted.
  bool isComplementary(Value other) const {
    return value.getPointer() == other && value.getInt();
  }

  bool isComplementary(const InvertibleValue &other) const {
    return value == other.value && value.getInt() != other.value.getInt();
  }

  /// Returns true if the value is the same as the other value and not inverted.
  bool isEquivalent(Value other) const {
    return value.getPointer() == other && !value.getInt();
  }

  bool isEquivalent(const InvertibleValue &other) const {
    return value == other.value;
  }

  bool isInverted() const { return value.getInt(); }
  Value getValue() const { return value.getPointer(); }
};

/// Helper struct to represent a value that may be inverted and has a depth.
struct TimedInvertibleValue : InvertibleValue {
  int64_t depth;
  TimedInvertibleValue(Value value, bool inverted, int64_t depth)
      : InvertibleValue(value, inverted), depth(depth) {}
  bool operator<(const TimedInvertibleValue &other) const {
    return depth < other.depth;
  }
};

/// Helper struct to represent values that may be inverted and have a depth.
struct OrderedValues {
  SmallVector<TimedInvertibleValue, 3> invertibleValues;

  OrderedValues(OperandRange operands, ArrayRef<bool> inversions,
                ArrayRef<int64_t> depths);

  static FailureOr<OrderedValues> get(mig::MajorityInverterOp op,
                                      IncrementalLongestPathAnalysis *analysis);

  void dump(llvm::raw_ostream &os) const {
    for (size_t i = 0; i < invertibleValues.size(); ++i) {
      os << "  Child " << i << ": " << invertibleValues[i].value.getPointer()
         << " (inverted: " << invertibleValues[i].value.getInt()
         << ", depth: " << invertibleValues[i].depth << ")\n";
    }
  }

  Value getValue(size_t idx) const { return invertibleValues[idx].getValue(); }
  bool isInverted(size_t idx) const {
    return invertibleValues[idx].isInverted();
  }
  int64_t getDepth(size_t idx) const { return invertibleValues[idx].depth; }

  auto &operator[](size_t idx) { return invertibleValues[idx]; }
  auto operator[](size_t idx) const { return invertibleValues[idx]; }
};
} // namespace synth
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/Synth/Synth.h.inc"

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_H
