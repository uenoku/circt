//===- TimingAnalysis.cpp - Two-Stage Timing Analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>

#define DEBUG_TYPE "timing-analysis"

using namespace circt;
using namespace circt::synth::timing;

static void
updateAdaptiveDamping(TimingAnalysisOptions::AdaptiveSlewHintDampingMode mode,
                      double maxDelta, double previousDelta, double &damping) {
  switch (mode) {
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Disabled:
    return;
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Conservative:
    if (maxDelta > previousDelta * 0.99)
      damping = std::max(0.2, damping * 0.75);
    else if (maxDelta < previousDelta * 0.7)
      damping = std::min(1.0, damping * 1.05);
    return;
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive:
    if (maxDelta > previousDelta * 0.97)
      damping = std::max(0.1, damping * 0.5);
    else if (maxDelta < previousDelta * 0.6)
      damping = std::min(1.0, damping * 1.2);
    return;
  }
}

//===----------------------------------------------------------------------===//
// TimingAnalysis Implementation
//===----------------------------------------------------------------------===//

TimingAnalysis::TimingAnalysis(hw::HWModuleOp module,
                               TimingAnalysisOptions options)
    : module(module), options(std::move(options)) {}

TimingAnalysis::TimingAnalysis(mlir::ModuleOp moduleOp,
                               hw::HWModuleOp topModule,
                               TimingAnalysisOptions options, bool hierarchical)
    : moduleOp(moduleOp), module(topModule), hierarchical(hierarchical),
      options(std::move(options)) {}

TimingAnalysis::~TimingAnalysis() = default;

std::unique_ptr<TimingAnalysis>
TimingAnalysis::create(hw::HWModuleOp module, TimingAnalysisOptions options) {
  return std::unique_ptr<TimingAnalysis>(
      new TimingAnalysis(module, std::move(options)));
}

std::unique_ptr<TimingAnalysis>
TimingAnalysis::create(mlir::ModuleOp moduleOp, StringRef topModuleName,
                       TimingAnalysisOptions options) {
  if (!moduleOp || topModuleName.empty())
    return nullptr;

  auto topModule = moduleOp.lookupSymbol<hw::HWModuleOp>(topModuleName);
  if (!topModule)
    return nullptr;

  return std::unique_ptr<TimingAnalysis>(new TimingAnalysis(
      moduleOp, topModule, std::move(options), /*hierarchical=*/true));
}

LogicalResult TimingAnalysis::buildGraph() {
  LLVM_DEBUG(llvm::dbgs() << "Building timing graph...\n");

  if (hierarchical)
    graph = std::make_unique<TimingGraph>(moduleOp, module);
  else
    graph = std::make_unique<TimingGraph>(module);
  return graph->build(options.delayModel);
}

LogicalResult TimingAnalysis::runArrivalAnalysis() {
  return runArrivalAnalysisWithLoadSlewHints({});
}

LogicalResult
TimingAnalysis::runArrivalAnalysisWithLoadSlewHints(ArrayRef<double> hints) {
  if (!graph) {
    if (failed(buildGraph()))
      return failure();
  }

  ArrivalAnalysis::Options arrivalOpts;
  arrivalOpts.keepAllArrivals = options.keepAllArrivals;
  arrivalOpts.startPointPatterns.assign(options.startPointPatterns.begin(),
                                        options.startPointPatterns.end());
  arrivalOpts.initialSlew = options.initialSlew;
  arrivalOpts.loadSlewHints.assign(hints.begin(), hints.end());

  arrivals = std::make_unique<ArrivalAnalysis>(*graph, arrivalOpts,
                                               options.delayModel);
  if (failed(arrivals->run()))
    return failure();

  enumerator =
      std::make_unique<PathEnumerator>(*graph, *arrivals, options.delayModel);
  return success();
}

LogicalResult TimingAnalysis::runBackwardAnalysis() {
  if (!arrivals) {
    if (failed(runArrivalAnalysis()))
      return failure();
  }

  RequiredTimeAnalysis::Options ratOpts;
  ratOpts.clockPeriod = options.clockPeriod;

  requiredTimeAnalysis = std::make_unique<RequiredTimeAnalysis>(
      *graph, *arrivals, ratOpts, options.delayModel);
  return requiredTimeAnalysis->run();
}

LogicalResult TimingAnalysis::runFullAnalysis() {
  if (failed(buildGraph()))
    return failure();

  lastArrivalIterations = 0;
  lastArrivalConverged = true;
  lastMaxSlewDelta = 0.0;
  lastRelativeSlewDelta = 0.0;
  lastEffectiveSlewConvergenceRelativeEpsilon =
      options.slewConvergenceRelativeEpsilon;
  lastWaveformCoupledConvergence = false;
  lastSlewDeltaHistory.clear();
  lastSlewDampingHistory.clear();
  lastAppliedSlewHintDamping = std::clamp(options.slewHintDamping, 0.0, 1.0);
  if (options.delayModel && options.delayModel->usesSlewPropagation()) {
    bool waveformCoupled = options.enableWaveformCoupledConvergence &&
                           options.delayModel->usesWaveformPropagation();
    lastWaveformCoupledConvergence = waveformCoupled;

    auto adaptiveMode = options.adaptiveSlewHintDampingMode;
    double relativeEpsilon = options.slewConvergenceRelativeEpsilon;
    double damping = lastAppliedSlewHintDamping;
    if (waveformCoupled) {
      if (adaptiveMode ==
          TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Disabled)
        adaptiveMode =
            TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Conservative;
      if (relativeEpsilon <= 0.0)
        relativeEpsilon = 0.05;
      damping = std::min(damping, 0.8);
    }
    lastEffectiveSlewConvergenceRelativeEpsilon = relativeEpsilon;

    double previousDelta = 0.0;
    double firstDelta = 0.0;
    bool hasPreviousDelta = false;
    SmallVector<double> previousSlews(graph->getNumNodes(),
                                      options.initialSlew);
    bool converged = false;

    unsigned maxIterations = std::max(1u, options.maxSlewIterations);
    if (waveformCoupled)
      maxIterations = std::max(maxIterations, 2u);
    for (unsigned iter = 0; iter < maxIterations; ++iter) {
      if (failed(runArrivalAnalysisWithLoadSlewHints(previousSlews)))
        return failure();
      ++lastArrivalIterations;

      double maxDelta = 0.0;
      for (const auto &node : graph->getNodes()) {
        auto id = node->getId().index;
        double current = arrivals->getMaxArrivalSlew(node.get());
        maxDelta = std::max(maxDelta, std::abs(current - previousSlews[id]));
      }
      lastSlewDeltaHistory.push_back(maxDelta);
      lastSlewDampingHistory.push_back(damping);
      lastMaxSlewDelta = maxDelta;
      if (iter == 0)
        firstDelta = maxDelta;
      lastRelativeSlewDelta = firstDelta > 0.0 ? (maxDelta / firstDelta) : 0.0;

      bool convergedAbsolute = maxDelta <= options.slewConvergenceEpsilon;
      bool convergedRelative =
          relativeEpsilon > 0.0 && lastRelativeSlewDelta <= relativeEpsilon;
      bool allowConvergenceNow = !waveformCoupled || iter > 0;
      if (allowConvergenceNow && (convergedAbsolute || convergedRelative)) {
        converged = true;
        lastAppliedSlewHintDamping = damping;
        break;
      }

      if (hasPreviousDelta)
        updateAdaptiveDamping(adaptiveMode, maxDelta, previousDelta, damping);
      previousDelta = maxDelta;
      hasPreviousDelta = true;
      lastAppliedSlewHintDamping = damping;

      for (const auto &node : graph->getNodes()) {
        auto id = node->getId().index;
        double current = arrivals->getMaxArrivalSlew(node.get());
        previousSlews[id] =
            previousSlews[id] + damping * (current - previousSlews[id]);
      }
    }

    lastArrivalConverged = converged;
  } else {
    if (failed(runArrivalAnalysis()))
      return failure();
    lastArrivalIterations = 1;
    lastArrivalConverged = true;
    lastMaxSlewDelta = 0.0;
    lastRelativeSlewDelta = 0.0;
    lastSlewDeltaHistory.push_back(0.0);
    lastSlewDampingHistory.push_back(
        std::clamp(options.slewHintDamping, 0.0, 1.0));
    lastAppliedSlewHintDamping = std::clamp(options.slewHintDamping, 0.0, 1.0);
  }

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
TimingAnalysis::getKWorstPaths(size_t k, SmallVectorImpl<TimingPath> &results) {
  if (!enumerator) {
    if (failed(runArrivalAnalysis()))
      return failure();
  }
  return enumerator->getKWorstPaths(k, results);
}

LogicalResult TimingAnalysis::getPathsTo(ArrayRef<std::string> patterns,
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
