//===- TimingAnalysis.h - Two-Stage Timing Analysis Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main TimingAnalysis interface that provides two-stage
// static timing analysis similar to commercial EDA tools:
//
// Stage 1 (Graph-Based Analysis):
//   - Build timing graph from IR
//   - Forward propagation: compute arrival times at all nodes
//   - (Optional) Backward propagation: compute required times
//
// Stage 2 (Path-Based Analysis):
//   - On-demand path enumeration based on queries
//   - Support for -from, -to, -through constraints
//   - K-worst paths enumeration
//
// Usage:
//   auto analysis = TimingAnalysis::create(module, options);
//   analysis->runArrivalAnalysis();
//
//   PathQuery query;
//   query.fromPatterns = {"*cmd*"};
//   query.toPatterns = {"*out*"};
//   SmallVector<TimingPath> paths;
//   analysis->enumeratePaths(query, paths);
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H

#include "circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h"
#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "circt/Dialect/Synth/Analysis/Timing/PathEnumerator.h"
#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"
#include <memory>

namespace circt {
namespace synth {
namespace timing {

//===----------------------------------------------------------------------===//
// TimingAnalysisOptions
//===----------------------------------------------------------------------===//

/// Options for configuring the timing analysis.
struct TimingAnalysisOptions {
  /// If true, keep arrival times from all start points at each node.
  /// This allows querying which start points contribute to each node.
  /// More memory intensive but enables richer queries.
  bool keepAllArrivals = false;

  /// If set, only analyze paths from start points matching these patterns.
  /// This can significantly reduce analysis time and memory.
  SmallVector<std::string, 2> startPointPatterns;
};

//===----------------------------------------------------------------------===//
// TimingAnalysis - Main interface
//===----------------------------------------------------------------------===//

/// The main interface for two-stage static timing analysis.
class TimingAnalysis {
public:
  /// Create a timing analysis for a module.
  static std::unique_ptr<TimingAnalysis>
  create(hw::HWModuleOp module, TimingAnalysisOptions options = {});

  ~TimingAnalysis();

  // Non-copyable
  TimingAnalysis(const TimingAnalysis &) = delete;
  TimingAnalysis &operator=(const TimingAnalysis &) = delete;

  //===--------------------------------------------------------------------===//
  // Stage 1: Graph-Based Analysis
  //===--------------------------------------------------------------------===//

  /// Build the timing graph from the module.
  LogicalResult buildGraph();

  /// Run arrival time propagation (forward analysis).
  LogicalResult runArrivalAnalysis();

  /// Check if the graph has been built.
  bool hasGraph() const { return graph != nullptr; }

  /// Check if arrival analysis has been run.
  bool hasArrivalData() const { return arrivals != nullptr; }

  /// Get the timing graph.
  const TimingGraph &getGraph() const { return *graph; }

  /// Get arrival analysis results.
  const ArrivalAnalysis &getArrivals() const { return *arrivals; }

  //===--------------------------------------------------------------------===//
  // Stage 2: Path-Based Analysis
  //===--------------------------------------------------------------------===//

  /// Enumerate paths matching the query.
  LogicalResult enumeratePaths(const PathQuery &query,
                               SmallVectorImpl<TimingPath> &results);

  /// Get paths from specific start points to specific end points.
  LogicalResult getPaths(ArrayRef<std::string> fromPatterns,
                         ArrayRef<std::string> toPatterns,
                         SmallVectorImpl<TimingPath> &results,
                         size_t maxPaths = 0);

  /// Get K worst paths in the design.
  LogicalResult getKWorstPaths(size_t k, SmallVectorImpl<TimingPath> &results);

  /// Get paths to a specific object by name pattern.
  LogicalResult getPathsTo(ArrayRef<std::string> patterns,
                           SmallVectorImpl<TimingPath> &results);

  /// Get paths from a specific object by name pattern.
  LogicalResult getPathsFrom(ArrayRef<std::string> patterns,
                             SmallVectorImpl<TimingPath> &results);

  //===--------------------------------------------------------------------===//
  // Object Collection API (like EDA tool commands)
  //===--------------------------------------------------------------------===//

  /// Get nodes matching patterns (like get_cells/get_pins).
  ObjectCollection getObjects(ArrayRef<std::string> patterns);

  /// Get all start points (like all_registers -output_pins).
  ObjectCollection getAllStartPoints();

  /// Get all end points (like all_registers -input_pins).
  ObjectCollection getAllEndPoints();

  //===--------------------------------------------------------------------===//
  // Query API
  //===--------------------------------------------------------------------===//

  /// Get arrival time at a node.
  int64_t getArrivalTime(const TimingNode *node) const;

  /// Get arrival time at a node by name.
  int64_t getArrivalTime(StringRef nodeName) const;

private:
  explicit TimingAnalysis(hw::HWModuleOp module, TimingAnalysisOptions options);

  hw::HWModuleOp module;
  TimingAnalysisOptions options;

  std::unique_ptr<TimingGraph> graph;
  std::unique_ptr<ArrivalAnalysis> arrivals;
  std::unique_ptr<PathEnumerator> enumerator;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H

