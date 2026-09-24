//===- TimingResult.h - Structured timing result model --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Backend-neutral timing result: whole-design endpoint aggregates with
// documented normalization semantics (WNS/TNS/failing endpoints), coverage
// accounting, and bounded worst-path traces. See
// docs/StaticTimingAnalysisLibraryIntegration.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TIMING_TIMINGRESULT_H
#define CIRCT_DIALECT_SYNTH_TIMING_TIMINGRESULT_H

#include "circt/Dialect/Synth/Timing/TimingTypes.h"

#include <cstdint>
#include <optional>

namespace circt {
namespace synth {
namespace timing {

/// Whether the analysis covered everything requested.
enum class TimingResultStatus { Complete, Incomplete, Failed };

/// One transition of one endpoint. Arrival and required are present only
// when the backend traced the endpoint (top-K paths); slack is always
// present for analyzed endpoints. This keeps full-coverage aggregates
// affordable while preserving detailed traces where they matter.
struct TimingTransitionResult {
  TimingTransition transition = TimingTransition::Rise;
  std::optional<TimePs> arrival;
  std::optional<TimePs> required;
  TimePs slack;
  std::optional<TimingObjectKey> worstLaunch;
};

/// One endpoint with all analyzed transitions and the normalized worst.
struct TimingEndpointResult {
  TimingObjectKey endpoint;
  mlir::StringAttr pathGroup;
  SmallVector<TimingTransitionResult, 2> transitions;
  TimePs worstSlack;
};

/// One point of a path trace.
struct TimingPathPoint {
  TimingObjectKey object;
  TimingTransition transition = TimingTransition::Rise;
  TimePs arrival;
};

/// One worst-path trace.
struct TimingPathResult {
  TimingObjectKey endpoint;
  TimingTransition transition = TimingTransition::Rise;
  TimePs arrival;
  TimePs required;
  TimePs slack;
  SmallVector<TimingPathPoint, 2> points;
};

/// Endpoint accounting. A Complete result has every expected endpoint either
/// analyzed or explicitly classified by a permitted exclusion.
struct TimingCoverage {
  uint64_t expectedEndpoints = 0;
  uint64_t constrainedEndpoints = 0;
  uint64_t analyzedEndpoints = 0;
  uint64_t unconstrainedEndpoints = 0;
  uint64_t unsupportedEndpoints = 0;
};

/// A non-fatal diagnostic with an optional related object.
struct TimingDiagnostic {
  mlir::StringAttr message;
  std::optional<TimingObjectKey> object;
};

/// The complete timing result for one analysis request.
struct TimingResult {
  TimingResultStatus status = TimingResultStatus::Failed;
  mlir::StringAttr backend;
  mlir::StringAttr backendVersion;
  TimingAnalysisKind analysis = TimingAnalysisKind::SetupMax;
  std::optional<TimePs> wns;
  std::optional<TimePs> tns;
  uint64_t failingEndpoints = 0;
  TimingCoverage coverage;
  SmallVector<TimingEndpointResult, 2> endpoints;
  SmallVector<TimingPathResult, 2> paths;
  SmallVector<TimingDiagnostic> diagnostics;
};

/// Normalize aggregates per the documented semantics: each endpoint's worst
/// slack is the minimum over its transitions; WNS is the minimum endpoint
/// slack; TNS sums min(endpointSlack, 0) once per endpoint; failing counts
/// negative-slack endpoints. Fills wns/tns/failingEndpoints and each
/// endpoint's worstSlack. Absent when there are no endpoints.
void normalizeTimingResult(TimingResult &result);

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TIMING_TIMINGRESULT_H
