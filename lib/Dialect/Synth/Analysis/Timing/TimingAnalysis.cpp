//===- TimingAnalysis.cpp - Two-Stage Timing Analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "timing-analysis"

using namespace circt;
using namespace circt::synth::timing;

//===----------------------------------------------------------------------===//
// TimingAnalysis Implementation
//===----------------------------------------------------------------------===//

TimingAnalysis::TimingAnalysis(hw::HWModuleOp module,
                               TimingAnalysisOptions options)
    : module(module), options(std::move(options)) {}

TimingAnalysis::~TimingAnalysis() = default;

std::unique_ptr<TimingAnalysis>
TimingAnalysis::create(hw::HWModuleOp module, TimingAnalysisOptions options) {
  return std::unique_ptr<TimingAnalysis>(
      new TimingAnalysis(module, std::move(options)));
}

LogicalResult TimingAnalysis::buildGraph() {
  LLVM_DEBUG(llvm::dbgs() << "Building timing graph...\n");

  graph = std::make_unique<TimingGraph>(module);
  return graph->build();
}

LogicalResult TimingAnalysis::runArrivalAnalysis() {
  if (!graph) {
    LLVM_DEBUG(llvm::dbgs() << "Graph not built, building first...\n");
    if (failed(buildGraph()))
      return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "Running arrival analysis...\n");

  ArrivalAnalysis::Options arrivalOpts;
  arrivalOpts.keepAllArrivals = options.keepAllArrivals;
  arrivalOpts.startPointPatterns.assign(options.startPointPatterns.begin(),
                                        options.startPointPatterns.end());

  arrivals = std::make_unique<ArrivalAnalysis>(*graph, arrivalOpts);
  if (failed(arrivals->run()))
    return failure();

  // Create enumerator for path queries
  enumerator = std::make_unique<PathEnumerator>(*graph, *arrivals);

  return success();
}

LogicalResult TimingAnalysis::enumeratePaths(const PathQuery &query,
                                             SmallVectorImpl<TimingPath> &results) {
  if (!enumerator) {
    LLVM_DEBUG(llvm::dbgs()
               << "Arrival analysis not run, running first...\n");
    if (failed(runArrivalAnalysis()))
      return failure();
  }

  return enumerator->enumerate(query, results);
}

LogicalResult TimingAnalysis::getPaths(ArrayRef<std::string> fromPatterns,
                                       ArrayRef<std::string> toPatterns,
                                       SmallVectorImpl<TimingPath> &results,
                                       size_t maxPaths) {
  PathQuery query;
  query.fromPatterns.assign(fromPatterns.begin(), fromPatterns.end());
  query.toPatterns.assign(toPatterns.begin(), toPatterns.end());
  query.maxPaths = maxPaths;
  return enumeratePaths(query, results);
}

LogicalResult
TimingAnalysis::getKWorstPaths(size_t k,
                               SmallVectorImpl<TimingPath> &results) {
  if (!enumerator) {
    if (failed(runArrivalAnalysis()))
      return failure();
  }
  return enumerator->getKWorstPaths(k, results);
}

LogicalResult
TimingAnalysis::getPathsTo(ArrayRef<std::string> patterns,
                           SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.toPatterns.assign(patterns.begin(), patterns.end());
  return enumeratePaths(query, results);
}

LogicalResult
TimingAnalysis::getPathsFrom(ArrayRef<std::string> patterns,
                             SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.fromPatterns.assign(patterns.begin(), patterns.end());
  return enumeratePaths(query, results);
}

ObjectCollection TimingAnalysis::getObjects(ArrayRef<std::string> patterns) {
  if (!graph)
    return ObjectCollection();
  return ObjectCollection::fromPatterns(*graph, patterns);
}

ObjectCollection TimingAnalysis::getAllStartPoints() {
  if (!graph)
    return ObjectCollection();
  return ObjectCollection::allStartPoints(*graph);
}

ObjectCollection TimingAnalysis::getAllEndPoints() {
  if (!graph)
    return ObjectCollection();
  return ObjectCollection::allEndPoints(*graph);
}

int64_t TimingAnalysis::getArrivalTime(const TimingNode *node) const {
  if (!arrivals)
    return 0;
  return arrivals->getMaxArrivalTime(node);
}

int64_t TimingAnalysis::getArrivalTime(StringRef nodeName) const {
  if (!graph || !arrivals)
    return 0;

  // Find node by name
  for (const auto &node : graph->getNodes()) {
    if (node->getName() == nodeName)
      return arrivals->getMaxArrivalTime(node.get());
  }
  return 0;
}

