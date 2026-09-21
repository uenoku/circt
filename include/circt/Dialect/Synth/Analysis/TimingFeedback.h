//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared timing-feedback records: parsing, validation, and endpoint
// resolution for OpenSTA-derived timing feedback consumed by probe-guided
// passes (`synth-import-timing-feedback`, refinement-region discovery) and
// the retry driver. Two formats are accepted:
//
// - versioned: {"schema": "circt.synth.timing-feedback.v1",
//               "time_unit": "ns", "top": "Top",
//               "paths": [{"startpoint": ..., "endpoint": ...,
//                          "required": 9.82, "arrival": 10.46,
//                          "slack": -0.64, "path_group": "clk"}]}
// - legacy flat list (sta_scoreboard --violators-out):
//   [{"startpoint": ..., "endpoint": ...,
//     "required_ns": 9.82, "arrival_ns": 10.46, "slack_ns": -0.64}]
//
// All values are converted once to picoseconds with checked rounding.
// In strict mode any invalid record fails parsing; in permissive mode
// invalid records are diagnosed through the callback and skipped.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGFEEDBACK_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGFEEDBACK_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace circt {
namespace synth {

// One validated timing path. Times are picoseconds; arrival and slack are
// optional (period-derived constraints may carry only a requirement).
struct TimingFeedbackPath {
  std::string startpoint;
  std::string endpoint;
  std::string pathGroup;
  int64_t requiredPs = 0;
  std::optional<int64_t> arrivalPs;
  std::optional<int64_t> slackPs;
};

// Parse result: the validated paths plus resolution statistics.
struct TimingFeedback {
  std::string top;
  std::vector<TimingFeedbackPath> paths;
  // Counters for diagnostics (records seen, skipped, and why).
  unsigned numRecords = 0;
  unsigned numResolved = 0;
  unsigned numUnresolved = 0;
  unsigned numInvalid = 0;
  unsigned numAmbiguous = 0;

  // Parse and validate feedback JSON. `expectedTop` (empty to skip) is
  // checked against the versioned schema's top field. In strict mode the
  // first invalid record is a failure; otherwise invalid records are
  // reported through `diagnose` and skipped. Returns failure only in
  // strict mode or when the document itself is malformed.
  static FailureOr<TimingFeedback>
  parse(StringRef jsonText, StringRef expectedTop, bool strict,
        llvm::function_ref<void(const Twine &)> diagnose);
};

// A resolved hierarchical endpoint: the flop instance and its D operand.
struct ResolvedEndpoint {
  hw::InstanceOp instance;
  unsigned dataOperandIdx = 0;
};

// Resolve a hierarchical endpoint ("u_core/u_alu/result_reg/D") to the
// mapped flop instance and D operand index. Fails when the path does not
// resolve to exactly one instance with a D input (missing, ambiguous, or
// clock-pin endpoints all fail).
FailureOr<ResolvedEndpoint> resolveFeedbackEndpoint(ModuleOp top,
                                                    StringRef endpoint);

// Resolve a hierarchical launch ("u_core/u_alu/a_reg/Q") to the flop Q
// value. Fails on missing/ambiguous/non-flop targets.
FailureOr<Value> resolveFeedbackLaunch(ModuleOp top, StringRef startpoint);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMINGFEEDBACK_H
