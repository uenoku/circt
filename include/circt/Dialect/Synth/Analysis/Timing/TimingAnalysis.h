//===- TimingAnalysis.h - Two-Stage Timing Analysis Interface ----*- C++
//-*-===//
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
//   - Backward propagation: compute required times and slack
//
// Stage 2 (Path-Based Analysis):
//   - On-demand path enumeration based on queries
//   - Support for -from, -to, -through constraints
//   - K-worst paths enumeration
//
// Usage:
//   auto analysis = TimingAnalysis::create(module, options);
//   analysis->runFullAnalysis();
//   analysis->reportTiming(llvm::outs());
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H

#include "circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h"
#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "circt/Dialect/Synth/Analysis/Timing/PathEnumerator.h"
#include "circt/Dialect/Synth/Analysis/Timing/RequiredTimeAnalysis.h"
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
  bool keepAllArrivals = false;

  /// If set, only analyze paths from start points matching these patterns.
  SmallVector<std::string, 2> startPointPatterns;

  /// Clock period constraint for backward analysis. 0 = unconstrained.
  int64_t clockPeriod = 0;

  /// Custom delay model. If null, uses default AIGLevelDelayModel.
  const DelayModel *delayModel = nullptr;
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

  /// Create a hierarchical timing analysis rooted at `topModuleName`.
  /// Returns nullptr if the top module is missing or invalid.
  static std::unique_ptr<TimingAnalysis>
  create(mlir::ModuleOp moduleOp, StringRef topModuleName,
         TimingAnalysisOptions options = {});

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

  /// Run backward required time analysis.
  LogicalResult runBackwardAnalysis();

  /// Run full analysis: build graph, forward, backward.
  LogicalResult runFullAnalysis();

  /// Check if the graph has been built.
  bool hasGraph() const { return graph != nullptr; }

  /// Check if arrival analysis has been run.
  bool hasArrivalData() const { return arrivals != nullptr; }

  /// Check if backward analysis has been run.
  bool hasRequiredTimeData() const { return requiredTimeAnalysis != nullptr; }

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

  /// Get paths with through-point constraint.
  LogicalResult getPaths(ArrayRef<std::string> fromPatterns,
                         ArrayRef<std::string> toPatterns,
                         ArrayRef<std::string> throughPatterns,
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
  // Object Collection API
  //===--------------------------------------------------------------------===//

  ObjectCollection getObjects(ArrayRef<std::string> patterns);
  ObjectCollection getAllStartPoints();
  ObjectCollection getAllEndPoints();

  //===--------------------------------------------------------------------===//
  // Query API
  //===--------------------------------------------------------------------===//

  /// Get arrival time at a node.
  int64_t getArrivalTime(const TimingNode *node) const;
  int64_t getArrivalTime(StringRef nodeName) const;

  /// Get required time at a node.
  int64_t getRequiredTime(const TimingNode *node) const;

  /// Get slack at a node.
  int64_t getSlack(const TimingNode *node) const;

  /// Get worst slack across all endpoints.
  int64_t getWorstSlack() const;

  //===--------------------------------------------------------------------===//
  // Reporting
  //===--------------------------------------------------------------------===//

  /// Print a timing report.
  void reportTiming(llvm::raw_ostream &os, size_t numPaths = 10);

private:
  explicit TimingAnalysis(hw::HWModuleOp module, TimingAnalysisOptions options);
  TimingAnalysis(mlir::ModuleOp moduleOp, hw::HWModuleOp topModule,
                 TimingAnalysisOptions options, bool hierarchical);

  mlir::ModuleOp moduleOp;
  hw::HWModuleOp module;
  bool hierarchical = false;
  TimingAnalysisOptions options;

  std::unique_ptr<TimingGraph> graph;
  std::unique_ptr<ArrivalAnalysis> arrivals;
  std::unique_ptr<PathEnumerator> enumerator;
  std::unique_ptr<RequiredTimeAnalysis> requiredTimeAnalysis;
};

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGANALYSIS_H
