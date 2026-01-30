//===- RequiredTimeAnalysis.h - Backward Propagation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RequiredTimeAnalysis class for backward required time
// propagation. Given arrival times and a clock period constraint, it computes
// required arrival times (RAT) and slack at every node.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_REQUIREDTIMEANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_REQUIREDTIMEANALYSIS_H

#include "circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h"
#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"

namespace circt {
namespace synth {
namespace timing {

/// Stores required arrival time for a single node.
class NodeRequiredData {
public:
  void setRequiredTime(int64_t rat) { requiredTime = rat; }
  int64_t getRequiredTime() const { return requiredTime; }
  bool isSet() const { return set; }
  void markSet() { set = true; }

private:
  int64_t requiredTime = 0;
  bool set = false;
};

/// Backward required time propagation through the timing graph.
class RequiredTimeAnalysis {
public:
  struct Options {
    /// Clock period constraint. 0 = unconstrained (RAT = max AT at endpoints).
    int64_t clockPeriod;
    Options() : clockPeriod(0) {}
  };

  RequiredTimeAnalysis(const TimingGraph &graph,
                       const ArrivalAnalysis &arrivals,
                       Options options = Options());

  /// Run the backward propagation.
  LogicalResult run();

  /// Get the required time at a node.
  int64_t getRequiredTime(TimingNodeId nodeId) const;
  int64_t getRequiredTime(const TimingNode *node) const {
    return getRequiredTime(node->getId());
  }

  /// Get the slack at a node (RAT - AT).
  int64_t getSlack(TimingNodeId nodeId) const;
  int64_t getSlack(const TimingNode *node) const {
    return getSlack(node->getId());
  }

  /// Get the worst (minimum) slack across all endpoints.
  int64_t getWorstSlack() const { return worstSlack; }

private:
  const TimingGraph &graph;
  const ArrivalAnalysis &arrivals;
  Options options;

  SmallVector<NodeRequiredData> requiredData;
  int64_t worstSlack = 0;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_REQUIREDTIMEANALYSIS_H
