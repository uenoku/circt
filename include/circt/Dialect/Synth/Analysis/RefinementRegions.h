//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Datapath refinement regions: deterministic partitioning of datapath logic
// into policy units, selection against timing pressure (STA feedback
// violations and structural criticality), and explicit per-operation policy
// records.
//
// A region is a connected component of datapath operations joined through
// transparent arithmetic glue (extracts, concats, constants). Muxes,
// comparisons, module boundaries, sequential elements, probes, and unknown
// operations are region boundaries. Regions are the unit of policy
// selection; the selected policy is recorded two ways:
//
// - in IR, as a `datapath.refine.policy` string attribute on member
//   datapath operations (and on comb add/compare roots whose operand cone
//   reaches a selected region), consumed by `datapath-reduce-delay` and
//   `convert-datapath-to-comb`; and
// - in a JSON decision report (stable region IDs), consumed by the retry
//   driver for ranking, manifests, and cycle detection.
//
// Region IDs (`@Module:preorder`) are deterministic for fixed input and
// pipeline flags; pointer values and SSA names never appear in them.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_REFINEMENTREGIONS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_REFINEMENTREGIONS_H

#include "circt/Dialect/Synth/Analysis/TimingFeedback.h"
#include "circt/Dialect/Synth/Analysis/TimingPressureAnalysis.h"
#include "llvm/ADT/MapVector.h"
#include <map>

namespace circt {
namespace synth {

// Transient per-operation policy record. Removed with the probes by
// `synth-strip-timing-probes`; never part of committed IR.
inline StringRef refinePolicyAttrName() { return "datapath.refine.policy"; }

enum class DatapathRefinementPolicy {
  Baseline,
  ReduceDelay,
  ForceBooth,
  CompressToAdd,
  FastestKnown
};

// Parse a policy name (baseline|reduce-delay|booth|compress-to-add|
// fastest|auto). auto resolves to ReduceDelay. Fails on unknown names.
FailureOr<DatapathRefinementPolicy> parseRefinementPolicy(StringRef name);
StringRef refinementPolicyName(DatapathRefinementPolicy policy);

// Read the recorded policy on an operation, if any.
std::optional<DatapathRefinementPolicy> getRecordedRefinePolicy(Operation *op);

// One selected (or explicitly baselined) datapath region.
struct RefinementRegion {
  std::string id;
  std::string module;
  SmallVector<Operation *> members;
  SmallVector<TimingProbeOp> probes;
  std::vector<std::string> probeIds;
  // Feedback endpoint strings constraining this region (violating paths).
  std::vector<std::string> endpoints;
  std::optional<int64_t> worstSlackLevels;
  TimingConfidence confidence = TimingConfidence::Unknown;
  bool violated = false;
  bool unresolvedBoundary = false;
  DatapathRefinementPolicy policy = DatapathRefinementPolicy::Baseline;
  std::string reason;
};

// Selection inputs.
struct RefinementConfig {
  // Structural pressure margin in levels; regions containing values with
  // slack >= -margin are pressured. Negative disables structural selection.
  int64_t pressureMarginLevels = 0;
  // Policy applied to selected regions (manifest entries override).
  DatapathRefinementPolicy defaultPolicy =
      DatapathRefinementPolicy::ReduceDelay;
  // Explicit per-region policies from a manifest (region ID -> policy).
  std::map<std::string, DatapathRefinementPolicy> manifest;
  // STA feedback overlay (violating endpoints select their cones).
  std::vector<TimingFeedbackPath> feedback;
  bool manifestStrict = false;
};

// Selection output: regions plus diagnostics.
struct RefinementPlan {
  std::vector<RefinementRegion> regions;
  // Regions named by the manifest that matched nothing.
  std::vector<std::string> unknownManifestIds;
  // Feedback endpoints that resolved to no probe.
  std::vector<std::string> unresolvedEndpoints;
  unsigned numFeedbackResolved = 0;
  unsigned numFeedbackViolated = 0;
};

// Partition datapath logic into regions and select policies. Pure analysis:
// attaches nothing. Use recordRefinementPlan to materialize.
RefinementPlan
discoverRefinementRegions(ModuleOp top, TimingPressureAnalysis &analysis,
                          const RefinementConfig &config,
                          function_ref<void(const Twine &)> diagnose);

// Attach the plan's selected policies as `datapath.refine.policy`
// attributes (member datapath ops plus reached comb add/compare roots).
// Returns the number of operations annotated.
unsigned recordRefinementPlan(const RefinementPlan &plan);

// Parse a policy manifest ({schema, regions:[{id, policy}]}). In strict
// mode unknown IDs or policies are failures; otherwise they are diagnosed
// and skipped (policies) or recorded (IDs).
FailureOr<std::map<std::string, DatapathRefinementPolicy>>
parsePolicyManifest(StringRef jsonText, bool strict,
                    function_ref<void(const Twine &)> diagnose);

// Serialize a plan as the versioned decision report.
std::string printRefinementReport(const RefinementPlan &plan,
                                  const TimingPressureAnalysis &analysis);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_REFINEMENTREGIONS_H
