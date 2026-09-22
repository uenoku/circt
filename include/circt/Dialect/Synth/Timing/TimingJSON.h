//===- TimingJSON.h - Timing result JSON serialization --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Versioned JSON reader/writer for TimingResult
// (schema "circt.synth.timing-result.v1", integral picoseconds,
// deterministic ordering). Uses LLVM JSON support; names are escaped
// correctly, never concatenated. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TIMING_TIMINGJSON_H
#define CIRCT_DIALECT_SYNTH_TIMING_TIMINGJSON_H

#include "circt/Dialect/Synth/Timing/TimingResult.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace synth {
namespace timing {

/// Result schema identifier. Bump on any incompatible change.
constexpr llvm::StringLiteral kTimingResultSchema =
    "circt.synth.timing-result.v1";

/// Parse versioned JSON into a result. Fails on schema mismatch or malformed
/// content; StringAttrs are interned in the given context.
mlir::LogicalResult parseTimingResult(llvm::StringRef json,
                                      TimingResult &result,
                                      mlir::MLIRContext *ctx);

/// Result schema identifier. Bump on any incompatible change.
constexpr llvm::StringLiteral kTimingFeedbackV2Schema =
    "circt.synth.timing-feedback.v2";

/// One v2 feedback path with structured endpoint keys. Times are
/// picoseconds; arrival and slack are optional (period-derived constraints
/// may carry only a requirement).
struct TimingFeedbackV2Path {
  TimingObjectKey startpoint;
  TimingObjectKey endpoint;
  TimePs required;
  std::optional<TimePs> arrival;
  std::optional<TimePs> slack;
  mlir::StringAttr pathGroup;
};

/// Parsed v2 feedback document.
struct TimingFeedbackV2 {
  mlir::StringAttr top;
  SmallVector<TimingFeedbackV2Path, 2> paths;
};

/// Serialize a timing result's endpoints as v2 feedback (integral
/// picoseconds, deterministic ordering). Emits every analyzed endpoint;
/// callers decide which need another refinement round.
void printTimingFeedbackV2(const TimingResult &result, llvm::StringRef top,
                           llvm::raw_ostream &os);

/// Parse a v2 feedback document. Fails on schema mismatch or malformed
/// content; StringAttrs are interned in the given context.
mlir::LogicalResult parseTimingFeedbackV2(llvm::StringRef json,
                                          TimingFeedbackV2 &feedback,
                                          mlir::MLIRContext *ctx);

/// Serialize with deterministic ordering (endpoints by key, paths by slack
/// then endpoint key, transition, and launch key).
void printTimingResult(const TimingResult &result, llvm::raw_ostream &os);

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TIMING_TIMINGJSON_H
