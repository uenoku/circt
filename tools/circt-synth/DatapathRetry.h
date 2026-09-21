//===- DatapathRetry.h - STA-judged datapath retry driver ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In-driver retry for datapath structural variants: run the pipeline with
// different datapath lowerings (Booth/compress/timing-aware/datapath-off),
// judge each candidate by OpenSTA slack (then area), and keep the best.
// Every candidate runs in a subprocess under a wall-clock timeout, so
// exploratory blowups (the hang trap) can never stall the loop.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SYNTH_DATAPATHRETRY_H
#define CIRCT_SYNTH_DATAPATHRETRY_H

#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace circt {
namespace synth {

struct DatapathRetryConfig {
  // Maximum structural variants to try after the baseline.
  int maxVariants = 0;
  // Satisfying slack threshold in ns (min-area among satisfying wins).
  double slackTarget = 0.0;
  // Per-candidate wall-clock timeout in seconds (0 disables).
  unsigned timeoutSecs = 600;
  bool keepTemps = false;
  // Probe-guided refinement rounds (scoped per-region candidates driven
  // by STA feedback). 0 disables driver rounds.
  bool refinement = false;
  int refinementIterations = 2;
  int pressureMarginLevels = 0;
  // Minimum WNS improvement in ps for a refinement round to continue.
  double minImprovementPs = 10.0;
  // Parent --timing-feedback file reused for round 0 (children otherwise
  // receive per-round generated feedback).
  std::string initialFeedback;
  // argv[0] of this driver (for self-exec and sibling lookup).
  std::string selfPath;
  // Full original argv (forwarded to children, minus retry flags).
  std::vector<std::string> argv;
  // Resolved flag values needed by the judge.
  std::string liberty; // comma-joined
  std::string top;
  std::string inputFile;
  std::string outputFile;    // "-" means stdout
  std::string userSdcOutput; // "" means none
  // Parent MLIR context for interning timing identity strings. Must be
  // set by callers that run candidates (never null in that path).
  mlir::MLIRContext *mlirContext = nullptr;
};

// Run the retry loop; writes the best candidate to outputFile (or stdout).
// Returns true on success.
bool runDatapathRetry(const DatapathRetryConfig &config);

} // namespace synth
} // namespace circt

#endif // CIRCT_SYNTH_DATAPATHRETRY_H
