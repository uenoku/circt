//===- ArrivalAnalysis.h - Arrival Time Propagation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ArrivalAnalysis class for forward arrival time
// propagation through the timing graph. This is the first stage of two-stage
// static timing analysis.
//
// The analysis computes:
// - Arrival time at every node from each contributing start point
// - Maximum arrival time at every node
// - Which start points contribute to each node's arrival time
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_ARRIVALANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_ARRIVALANALYSIS_H

#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace synth {
namespace timing {

//===----------------------------------------------------------------------===//
// ArrivalInfo - Arrival time information at a node
//===----------------------------------------------------------------------===//

/// Represents the arrival time at a node from a specific start point.
struct ArrivalInfo {
  TimingNodeId startPoint; // The start point this arrival came from
  int64_t arrivalTime;     // Arrival time from that start point
  double slew = 0.0;       // Slew at this node for that start point

  bool operator<(const ArrivalInfo &other) const {
    return arrivalTime < other.arrivalTime;
  }
};

//===----------------------------------------------------------------------===//
// NodeArrivalData - Complete arrival data for a node
//===----------------------------------------------------------------------===//

/// Stores all arrival time information for a single node.
/// Supports both "keep all arrivals" and "keep only max" modes.
class NodeArrivalData {
public:
  /// Add an arrival from a start point.
  void addArrival(TimingNodeId startPoint, int64_t arrivalTime,
                  double slew = 0.0);

  /// Get the maximum arrival time at this node.
  int64_t getMaxArrivalTime() const { return maxArrivalTime; }

  /// Get the start point that gives the maximum arrival time.
  TimingNodeId getMaxArrivalStartPoint() const { return maxStartPoint; }

  /// Get the slew associated with maximum arrival time.
  double getMaxArrivalSlew() const { return maxArrivalSlew; }

  /// Get all arrivals (when keepAllArrivals is true).
  ArrayRef<ArrivalInfo> getAllArrivals() const { return arrivals; }

  /// Check if this node has any arrivals.
  bool hasArrivals() const { return !arrivals.empty(); }

  /// Set whether to keep all arrivals or just the max.
  void setKeepAllArrivals(bool keep) { keepAllArrivals = keep; }

private:
  SmallVector<ArrivalInfo, 4> arrivals;
  int64_t maxArrivalTime = 0;
  double maxArrivalSlew = 0.0;
  TimingNodeId maxStartPoint;
  bool keepAllArrivals = false;
};

//===----------------------------------------------------------------------===//
// ArrivalAnalysis - Forward arrival time propagation
//===----------------------------------------------------------------------===//

/// Performs forward arrival time propagation through the timing graph.
/// This is the core of graph-based static timing analysis.
class ArrivalAnalysis {
public:
  struct Options {
    /// If true, keep arrival times from all start points at each node.
    /// If false, keep only the maximum arrival time (more memory efficient).
    bool keepAllArrivals = false;

    /// If set, only propagate from start points matching these patterns.
    SmallVector<std::string, 2> startPointPatterns;

    /// Initial slew for matched start points.
    double initialSlew = 0.0;

    Options() = default;
    Options(bool keepAll) : keepAllArrivals(keepAll) {}
    Options(bool keepAll, ArrayRef<std::string> patterns)
        : keepAllArrivals(keepAll),
          startPointPatterns(patterns.begin(), patterns.end()) {}
  };

  ArrivalAnalysis(const TimingGraph &graph);
  ArrivalAnalysis(const TimingGraph &graph, Options options,
                  const DelayModel *delayModel = nullptr);

  /// Run the arrival time propagation.
  LogicalResult run();

  /// Get arrival data for a node.
  const NodeArrivalData *getArrivalData(TimingNodeId nodeId) const;
  const NodeArrivalData *getArrivalData(const TimingNode *node) const {
    return getArrivalData(node->getId());
  }

  /// Get the maximum arrival time at a node.
  int64_t getMaxArrivalTime(TimingNodeId nodeId) const;
  int64_t getMaxArrivalTime(const TimingNode *node) const {
    return getMaxArrivalTime(node->getId());
  }

  /// Get all arrivals at a node (only valid if keepAllArrivals=true).
  ArrayRef<ArrivalInfo> getArrivals(TimingNodeId nodeId) const;

  /// Check if a start point contributes to a node's arrival.
  bool hasArrivalFrom(TimingNodeId nodeId, TimingNodeId startPoint) const;

  /// Get the timing graph.
  const TimingGraph &getGraph() const { return graph; }

  /// Get the matched start points (after pattern filtering).
  ArrayRef<TimingNode *> getMatchedStartPoints() const {
    return matchedStartPoints;
  }

private:
  /// Match start points against patterns.
  void matchStartPoints();

  /// Propagate arrivals through the graph.
  void propagate();

  const TimingGraph &graph;
  Options options;
  const DelayModel *delayModel = nullptr;

  /// Arrival data for each node, indexed by TimingNodeId.
  SmallVector<NodeArrivalData> arrivalData;

  /// Start points that matched the filter patterns.
  SmallVector<TimingNode *> matchedStartPoints;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_ARRIVALANALYSIS_H
