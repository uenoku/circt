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
  return graph->build(options.delayModel);
}

LogicalResult TimingAnalysis::runArrivalAnalysis() {
  if (!graph) {
    if (failed(buildGraph()))
      return failure();
  }

  ArrivalAnalysis::Options arrivalOpts;
  arrivalOpts.keepAllArrivals = options.keepAllArrivals;
  arrivalOpts.startPointPatterns.assign(options.startPointPatterns.begin(),
                                        options.startPointPatterns.end());

  arrivals = std::make_unique<ArrivalAnalysis>(*graph, arrivalOpts);
  if (failed(arrivals->run()))
    return failure();

  enumerator = std::make_unique<PathEnumerator>(*graph, *arrivals);
  return success();
}

LogicalResult TimingAnalysis::runBackwardAnalysis() {
  if (!arrivals) {
    if (failed(runArrivalAnalysis()))
      return failure();
  }

  RequiredTimeAnalysis::Options ratOpts;
  ratOpts.clockPeriod = options.clockPeriod;

  requiredTimeAnalysis =
      std::make_unique<RequiredTimeAnalysis>(*graph, *arrivals, ratOpts);
  return requiredTimeAnalysis->run();
}

LogicalResult TimingAnalysis::runFullAnalysis() {
  if (failed(buildGraph()))
    return failure();
  if (failed(runArrivalAnalysis()))
    return failure();
  return runBackwardAnalysis();
}

LogicalResult
TimingAnalysis::enumeratePaths(const PathQuery &query,
                               SmallVectorImpl<TimingPath> &results) {
  if (!enumerator) {
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

LogicalResult TimingAnalysis::getPaths(ArrayRef<std::string> fromPatterns,
                                       ArrayRef<std::string> toPatterns,
                                       ArrayRef<std::string> throughPatterns,
                                       SmallVectorImpl<TimingPath> &results,
                                       size_t maxPaths) {
  PathQuery query;
  query.fromPatterns.assign(fromPatterns.begin(), fromPatterns.end());
  query.toPatterns.assign(toPatterns.begin(), toPatterns.end());
  query.throughPatterns.assign(throughPatterns.begin(), throughPatterns.end());
  query.reconstructPaths = true; // Needed for through-point filtering
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
  for (const auto &node : graph->getNodes()) {
    if (node->getName() == nodeName)
      return arrivals->getMaxArrivalTime(node.get());
  }
  return 0;
}

int64_t TimingAnalysis::getRequiredTime(const TimingNode *node) const {
  if (!requiredTimeAnalysis)
    return 0;
  return requiredTimeAnalysis->getRequiredTime(node);
}

int64_t TimingAnalysis::getSlack(const TimingNode *node) const {
  if (!requiredTimeAnalysis)
    return 0;
  return requiredTimeAnalysis->getSlack(node);
}

int64_t TimingAnalysis::getWorstSlack() const {
  if (!requiredTimeAnalysis)
    return 0;
  return requiredTimeAnalysis->getWorstSlack();
}
