//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for exact synthesis of small Boolean truth tables.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
class OpBuilder;
class Value;
} // namespace mlir

namespace circt {
namespace synth {

/// Select which exact-synthesis node kinds may appear in the synthesized
/// implementation. Inversion is always available and is materialized with
/// `synth.aig.and_inv` when it cannot be absorbed into another node.
struct ExactSynthesisPolicy {
  bool allowAnd = true;
  bool allowXor = true;
  bool allowDot = true;
};

/// Exact-synthesize a single-output Boolean truth table into Synth IR.
///
/// `truthTable` must contain exactly `2^operands.size()` bits. `operands` are
/// ordered least-significant input first to match the truth-table bit indexing
/// used by `llvm::APInt`.
llvm::FailureOr<mlir::Value>
ExactSynthesis(mlir::OpBuilder &builder, llvm::APInt truthTable,
               llvm::ArrayRef<mlir::Value> operands,
               const ExactSynthesisPolicy &policy = {});

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_EXACTSYNTHESIS_H
