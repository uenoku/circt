//===- TimingResult.cpp - Timing aggregate normalization -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingResult.h"

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timing;

void circt::synth::timing::normalizeTimingResult(TimingResult &result) {
  if (result.endpoints.empty()) {
    result.wns.reset();
    result.tns.reset();
    result.failingEndpoints = 0;
    return;
  }
  int64_t wns = std::numeric_limits<int64_t>::max();
  int64_t tns = 0;
  uint64_t failing = 0;
  for (auto &endpoint : result.endpoints) {
    int64_t worst = std::numeric_limits<int64_t>::max();
    for (const auto &transition : endpoint.transitions)
      worst = std::min(worst, transition.slack.value);
    // An endpoint with no analyzed transitions carries no slack; it must
    // have been classified in coverage instead. Leave worst untouched
    // only when there is nothing to normalize.
    if (endpoint.transitions.empty())
      continue;
    endpoint.worstSlack.value = worst;
    wns = std::min(wns, worst);
    if (worst < 0) {
      tns += worst;
      ++failing;
    }
  }
  if (wns == std::numeric_limits<int64_t>::max()) {
    result.wns.reset();
    result.tns.reset();
    result.failingEndpoints = 0;
    return;
  }
  result.wns = TimePs{wns};
  result.tns = TimePs{tns};
  result.failingEndpoints = failing;
}
