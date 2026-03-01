//===- ArrivalAnalysis.cpp - Arrival Time Propagation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/ArrivalAnalysis.h"
#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arrival-analysis"

using namespace circt;
using namespace circt::synth::timing;

static DelayResult getArcDelay(const TimingArc *arc,
                               const DelayModel *delayModel, double inputSlew) {
  if (!delayModel || !arc->getOp())
    return {arc->getDelay(), inputSlew};

  DelayContext ctx;
  ctx.op = arc->getOp();
  ctx.inputValue = arc->getInputValue();
  ctx.outputValue = arc->getOutputValue();
  ctx.inputIndex = arc->getInputIndex();
  ctx.outputIndex = arc->getOutputIndex();
  ctx.inputSlew = inputSlew;

  auto result = delayModel->computeDelay(ctx);
  if (!delayModel->usesSlewPropagation())
    result.outputSlew = inputSlew;
  return result;
}

//===----------------------------------------------------------------------===//
// NodeArrivalData Implementation
//===----------------------------------------------------------------------===//

void NodeArrivalData::addArrival(TimingNodeId startPoint, int64_t arrivalTime,
                                 double slew) {
  // Update max if this is a new maximum
  if (arrivals.empty() || arrivalTime > maxArrivalTime ||
      (arrivalTime == maxArrivalTime && slew > maxArrivalSlew)) {
    maxArrivalTime = arrivalTime;
    maxArrivalSlew = slew;
    maxStartPoint = startPoint;
  }

  // Store all arrivals if requested
  if (keepAllArrivals) {
    // Check if we already have an arrival from this start point
    for (auto &arrival : arrivals) {
      if (arrival.startPoint == startPoint) {
        // Keep the maximum
        if (arrivalTime > arrival.arrivalTime ||
            (arrivalTime == arrival.arrivalTime && slew > arrival.slew)) {
          arrival.arrivalTime = arrivalTime;
          arrival.slew = slew;
        }
        return;
      }
    }
    arrivals.push_back({startPoint, arrivalTime, slew});
  } else {
    // Just keep the max
    if (arrivals.empty())
      arrivals.push_back({startPoint, arrivalTime, slew});
    else if (arrivalTime > arrivals[0].arrivalTime)
      arrivals[0] = {startPoint, arrivalTime, slew};
  }
}

//===----------------------------------------------------------------------===//
// ArrivalAnalysis Implementation
//===----------------------------------------------------------------------===//

ArrivalAnalysis::ArrivalAnalysis(const TimingGraph &graph)
    : ArrivalAnalysis(graph, Options()) {}

ArrivalAnalysis::ArrivalAnalysis(const TimingGraph &graph, Options options,
                                 const DelayModel *delayModel)
    : graph(graph), options(std::move(options)), delayModel(delayModel) {
  // Pre-allocate arrival data for all nodes
  arrivalData.resize(graph.getNumNodes());

  // Set keepAllArrivals flag on all NodeArrivalData
  for (auto &data : arrivalData)
    data.setKeepAllArrivals(this->options.keepAllArrivals);
}

LogicalResult ArrivalAnalysis::run() {
  LLVM_DEBUG(llvm::dbgs() << "Running arrival analysis\n");

  // Match start points against patterns
  matchStartPoints();

  if (matchedStartPoints.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No matching start points found\n");
    return success();
  }

  LLVM_DEBUG(llvm::dbgs() << "Matched " << matchedStartPoints.size()
                          << " start points\n");

  // Propagate arrivals through the graph
  propagate();

  return success();
}

void ArrivalAnalysis::matchStartPoints() {
  matchedStartPoints.clear();

  if (options.startPointPatterns.empty()) {
    // No patterns - use all start points
    for (auto *node : graph.getStartPoints())
      matchedStartPoints.push_back(node);
    return;
  }

  // Create matcher and filter start points
  auto matcherOrErr = ObjectMatcher::create(options.startPointPatterns);
  if (!matcherOrErr) {
    llvm::consumeError(matcherOrErr.takeError());
    return;
  }

  auto &matcher = *matcherOrErr;
  for (auto *node : graph.getStartPoints()) {
    if (matcher.matches(node->getName()))
      matchedStartPoints.push_back(node);
  }
}

void ArrivalAnalysis::propagate() {
  // Initialize arrival times at start points
  for (auto *startNode : matchedStartPoints) {
    auto &data = arrivalData[startNode->getId().index];
    data.addArrival(startNode->getId(), 0, options.initialSlew);
  }

  // Forward propagation in topological order
  for (auto *node : graph.getTopologicalOrder()) {
    auto &nodeData = arrivalData[node->getId().index];

    // For each fanout arc, propagate arrival time
    for (auto *arc : node->getFanout()) {
      auto *successor = arc->getTo();
      auto &succData = arrivalData[successor->getId().index];

      // Propagate all arrivals from this node
      for (const auto &arrival : nodeData.getAllArrivals()) {
        auto delay = getArcDelay(arc, delayModel, arrival.slew);
        int64_t newArrival = arrival.arrivalTime + delay.delay;
        succData.addArrival(arrival.startPoint, newArrival, delay.outputSlew);
      }
    }
  }

  LLVM_DEBUG({
    size_t nodesWithArrivals = 0;
    for (const auto &data : arrivalData) {
      if (data.hasArrivals())
        ++nodesWithArrivals;
    }
    llvm::dbgs() << "Arrival propagation complete: " << nodesWithArrivals
                 << " nodes have arrivals\n";
  });
}

const NodeArrivalData *
ArrivalAnalysis::getArrivalData(TimingNodeId nodeId) const {
  if (nodeId.index >= arrivalData.size())
    return nullptr;
  return &arrivalData[nodeId.index];
}

int64_t ArrivalAnalysis::getMaxArrivalTime(TimingNodeId nodeId) const {
  if (auto *data = getArrivalData(nodeId))
    return data->getMaxArrivalTime();
  return 0;
}

ArrayRef<ArrivalInfo> ArrivalAnalysis::getArrivals(TimingNodeId nodeId) const {
  if (auto *data = getArrivalData(nodeId))
    return data->getAllArrivals();
  return {};
}

bool ArrivalAnalysis::hasArrivalFrom(TimingNodeId nodeId,
                                     TimingNodeId startPoint) const {
  auto arrivals = getArrivals(nodeId);
  return llvm::any_of(arrivals, [&](const ArrivalInfo &info) {
    return info.startPoint == startPoint;
  });
}
