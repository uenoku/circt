//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Timing-pressure analysis: probe-transparent forward arrival and backward
// required-time propagation over combinational structure, used to focus
// datapath refinement on timing-pressured regions.
//
// Two layers, deliberately separated:
//
// - Structural reachability (delay-independent, hierarchical): which values
//   can reach which `synth.timing_probe` operations. Powers cone membership
//   (including STA-feedback-violating cones) even where delay modeling
//   gives up.
// - Level timing (structural levels, hierarchical summaries): forward
//   arrival in gate levels from launches (primary inputs, flop outputs,
//   constants) and backward required levels seeded from probes. Slack in
//   levels ranks candidate regions; it is NOT physical slack and must
//   never be presented as such (see TimingConfidence).
//
// Delay model (MVP, ranking-only): every combinational dataflow operation
// costs one level; probes, wiring (extract/concat/replicate), and constants
// cost zero. Unknown or unsupported operations terminate propagation and
// are counted, never invented. Shared module definitions use the tightest
// requirement across instance contexts (conservative).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGPRESSUREANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGPRESSUREANALYSIS_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <cstdint>
#include <optional>

namespace circt {
namespace synth {

// How much a timing point may be trusted. Only STA feedback may be
// presented as measured post-map slack.
enum class TimingConfidence {
  Unknown,
  StructuralEstimate,
  LibraryEstimate,
  STAFeedback,
};

// Per-value timing: structural levels plus an optional STA overlay handled
// by consumers (this analysis never parses timers itself).
struct TimingPoint {
  std::optional<int64_t> arrivalLevels;
  std::optional<int64_t> requiredLevels;
  std::optional<int64_t> slackLevels;
  TimingConfidence confidence = TimingConfidence::Unknown;
};

class TimingPressureAnalysis {
public:
  // Per-module tables (shared definitions hold the tightest context).
  struct ModuleTable {
    DenseMap<Value, int64_t> arrival;
    DenseMap<Value, int64_t> required;
    DenseSet<Value> reaches;
    DenseMap<Value, SmallVector<TimingProbeOp>> probes;
  };

  explicit TimingPressureAnalysis(Operation *op);

  // The analyzed scope (a ModuleOp or hw.module).
  Operation *getOperation() const { return scope; }

  // Per-value timing point (module-local value; hierarchical summaries
  // are already folded into shared-module tables).
  TimingPoint getPoint(Value value) const;

  // True when a forward structural walk from `value` reaches any probe.
  bool reachesProbe(Value value) const;

  // Probes constraining `value`, sorted deterministically
  // (module name, preorder index).
  SmallVector<TimingProbeOp> getConstrainingProbes(Value value) const;

  // True when `value` reaches a probe and its structural slack is within
  // `marginLevels` of critical (slack >= -margin). Margin 0 selects only
  // structurally critical values.
  bool isPressured(Value value, int64_t marginLevels) const;

  // Worst (minimum) structural slack over an operation's results, if all
  // results have slack.
  std::optional<int64_t> getWorstSlackLevels(Operation *op) const;

  // Diagnostic counters.
  unsigned getNumValuesWithArrival() const { return numWithArrival; }
  unsigned getNumValuesWithRequired() const { return numWithRequired; }
  unsigned getNumValuesWithSlack() const { return numWithSlack; }
  unsigned getNumUnknownBoundaries() const { return numUnknownBoundaries; }
  unsigned getNumHierarchicalEdges() const { return numHierarchicalEdges; }
  unsigned getNumSharedContexts() const { return numSharedContexts; }

private:
  Operation *scope;
  DenseMap<Operation *, ModuleTable> tables;

  unsigned numWithArrival = 0;
  unsigned numWithRequired = 0;
  unsigned numWithSlack = 0;
  unsigned numUnknownBoundaries = 0;
  unsigned numHierarchicalEdges = 0;
  unsigned numSharedContexts = 0;

  void analyze();
};

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGPRESSUREANALYSIS_H
