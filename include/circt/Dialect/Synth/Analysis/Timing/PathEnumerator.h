//===- PathEnumerator.h - Path Enumeration Engine ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PathEnumerator class for on-demand timing path
// enumeration. This is the second stage of two-stage static timing analysis.
//
// The enumerator supports:
// - Querying paths between specific start/end collections
// - K-worst paths enumeration
// - Through-point constraints
// - Efficient path reconstruction using arrival time data
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_PATHENUMERATOR_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_PATHENUMERATOR_H

#include "circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h"
#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace synth {
namespace timing {

//===----------------------------------------------------------------------===//
// TimingPath - A complete timing path
//===----------------------------------------------------------------------===//

/// Represents a complete timing path from start to end point.
class TimingPath {
public:
  TimingPath() = default;
  TimingPath(TimingNode *startPoint, TimingNode *endPoint, int64_t delay)
      : startPoint(startPoint), endPoint(endPoint), delay(delay) {}

  TimingNode *getStartPoint() const { return startPoint; }
  TimingNode *getEndPoint() const { return endPoint; }
  int64_t getDelay() const { return delay; }

  /// Get the intermediate nodes in this path (lazily computed).
  ArrayRef<TimingNode *> getIntermediateNodes() const { return intermediate; }

  /// Set intermediate nodes (for path reconstruction).
  void setIntermediateNodes(ArrayRef<TimingNode *> nodes) {
    intermediate.assign(nodes.begin(), nodes.end());
  }

  /// Print the path.
  void print(llvm::raw_ostream &os) const;

private:
  TimingNode *startPoint = nullptr;
  TimingNode *endPoint = nullptr;
  int64_t delay = 0;
  SmallVector<TimingNode *, 8> intermediate;
};

//===----------------------------------------------------------------------===//
// PathQuery - Specification for path enumeration
//===----------------------------------------------------------------------===//

/// Specifies what paths to enumerate.
struct PathQuery {
  /// Start point patterns (glob). Empty means all start points.
  SmallVector<std::string, 2> fromPatterns;

  /// End point patterns (glob). Empty means all end points.
  SmallVector<std::string, 2> toPatterns;

  /// Through point patterns (glob). Empty means no through constraint.
  SmallVector<std::string, 2> throughPatterns;

  /// Maximum number of paths to return. 0 means unlimited.
  size_t maxPaths = 0;

  /// If true, include paths to intermediate objects (not just endpoints).
  bool includeIntermediateEndpoints = false;

  /// If true, reconstruct full path (intermediate nodes).
  bool reconstructPaths = false;
};

//===----------------------------------------------------------------------===//
// PathEnumerator - On-demand path enumeration
//===----------------------------------------------------------------------===//

/// Enumerates timing paths based on queries. Uses pre-computed arrival time
/// data from ArrivalAnalysis for efficient enumeration.
class PathEnumerator {
public:
  PathEnumerator(const TimingGraph &graph, const ArrivalAnalysis &arrivals);

  /// Enumerate paths matching the query.
  LogicalResult enumerate(const PathQuery &query,
                          SmallVectorImpl<TimingPath> &results);

  /// Get paths to a specific node from all start points.
  LogicalResult getPathsToNode(TimingNode *node,
                               SmallVectorImpl<TimingPath> &results);

  /// Get paths from a specific start point to all endpoints.
  LogicalResult getPathsFromNode(TimingNode *startPoint,
                                 SmallVectorImpl<TimingPath> &results);

  /// Get the K worst (longest delay) paths.
  LogicalResult getKWorstPaths(size_t k, SmallVectorImpl<TimingPath> &results);

private:
  /// Match nodes against patterns.
  void matchNodes(ArrayRef<std::string> patterns,
                  SmallVectorImpl<TimingNode *> &matched,
                  bool startPointsOnly = false, bool endPointsOnly = false);

  /// Reconstruct the path from start to end.
  void reconstructPath(TimingPath &path);

  /// Check if a path goes through any of the specified nodes.
  bool goesThrough(const TimingPath &path, ArrayRef<TimingNode *> throughNodes);

  const TimingGraph &graph;
  const ArrivalAnalysis &arrivals;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_PATHENUMERATOR_H

