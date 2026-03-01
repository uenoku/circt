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

static double getArcInputCapacitance(const TimingArc *arc,
                                     const DelayModel *delayModel,
                                     double driverSlewHint) {
  if (!delayModel || !arc->getOp())
    return 0.0;

  DelayContext ctx;
  ctx.op = arc->getOp();
  ctx.inputValue = arc->getInputValue();
  ctx.outputValue = arc->getOutputValue();
  ctx.inputIndex = arc->getInputIndex();
  ctx.outputIndex = arc->getOutputIndex();
  ctx.inputSlew = driverSlewHint;
  return delayModel->getInputCapacitance(ctx);
}

static double getNodeOutputLoad(const TimingNode *node,
                                const DelayModel *delayModel,
                                ArrayRef<double> loadSlewHints) {
  if (!delayModel)
    return 0.0;
  double slewHint = 0.0;
  if (node->getId().index < loadSlewHints.size())
    slewHint = loadSlewHints[node->getId().index];
  double total = 0.0;
  for (auto *arc : node->getFanout())
    total += getArcInputCapacitance(arc, delayModel, slewHint);
  return total;
}

static DelayContext makeDelayContext(const TimingArc *arc, double inputSlew,
                                    double outputLoad,
                                    TransitionEdge outputEdge) {
  DelayContext ctx;
  ctx.op = arc->getOp();
  ctx.inputValue = arc->getInputValue();
  ctx.outputValue = arc->getOutputValue();
  ctx.inputIndex = arc->getInputIndex();
  ctx.outputIndex = arc->getOutputIndex();
  ctx.inputSlew = inputSlew;
  ctx.outputLoad = outputLoad;
  ctx.outputEdge = outputEdge;
  return ctx;
}

static DelayResult getArcDelay(const TimingArc *arc,
                               const DelayModel *delayModel, double inputSlew,
                               double outputLoad,
                               TransitionEdge outputEdge = TransitionEdge::Rise) {
  if (!delayModel || !arc->getOp())
    return {arc->getDelay(), inputSlew};

  auto ctx = makeDelayContext(arc, inputSlew, outputLoad, outputEdge);

  auto result = delayModel->computeDelay(ctx);
  if (!delayModel->usesSlewPropagation())
    result.outputSlew = inputSlew;
  return result;
}

static TimingSense getArcTimingSense(const TimingArc *arc,
                                     const DelayModel *delayModel) {
  if (!delayModel || !arc->getOp())
    return TimingSense::PositiveUnate;

  auto ctx = makeDelayContext(arc, 0.0, 0.0, TransitionEdge::Rise);
  return delayModel->getTimingSense(ctx);
}

//===----------------------------------------------------------------------===//
// NodeArrivalData Implementation
//===----------------------------------------------------------------------===//

void NodeArrivalData::addArrival(TimingNodeId startPoint, int64_t arrivalTime,
                                 double slew, TransitionEdge edge) {
  // Update max if this is a new maximum
  if (arrivals.empty() || arrivalTime > maxArrivalTime ||
      (arrivalTime == maxArrivalTime && slew > maxArrivalSlew)) {
    maxArrivalTime = arrivalTime;
    maxArrivalSlew = slew;
    maxStartPoint = startPoint;
  }

  // Update per-edge max tracking
  if (edge == TransitionEdge::Rise) {
    if (!hasRise || arrivalTime > maxRiseArrival) {
      maxRiseArrival = arrivalTime;
      maxRiseSlew = slew;
      hasRise = true;
    }
  } else {
    if (!hasFall || arrivalTime > maxFallArrival) {
      maxFallArrival = arrivalTime;
      maxFallSlew = slew;
      hasFall = true;
    }
  }

  // Store all arrivals if requested
  if (keepAllArrivals) {
    // Check if we already have an arrival from this start point with same edge
    for (auto &arrival : arrivals) {
      if (arrival.startPoint == startPoint && arrival.edge == edge) {
        // Keep the maximum
        if (arrivalTime > arrival.arrivalTime ||
            (arrivalTime == arrival.arrivalTime && slew > arrival.slew)) {
          arrival.arrivalTime = arrivalTime;
          arrival.slew = slew;
        }
        return;
      }
    }
    arrivals.push_back({startPoint, arrivalTime, slew, edge});
  } else {
    // Just keep the max
    if (arrivals.empty())
      arrivals.push_back({startPoint, arrivalTime, slew, edge});
    else if (arrivalTime > arrivals[0].arrivalTime)
      arrivals[0] = {startPoint, arrivalTime, slew, edge};
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

/// Determine output edge(s) given input edge and timing sense.
static SmallVector<TransitionEdge, 2>
getOutputEdges(TransitionEdge inputEdge, TimingSense sense) {
  switch (sense) {
  case TimingSense::PositiveUnate:
    return {inputEdge};
  case TimingSense::NegativeUnate:
    return {inputEdge == TransitionEdge::Rise ? TransitionEdge::Fall
                                              : TransitionEdge::Rise};
  case TimingSense::NonUnate:
    return {TransitionEdge::Rise, TransitionEdge::Fall};
  }
  return {inputEdge};
}

void ArrivalAnalysis::propagate() {
  // Initialize arrival times at start points with both rise and fall
  for (auto *startNode : matchedStartPoints) {
    auto &data = arrivalData[startNode->getId().index];
    data.addArrival(startNode->getId(), 0, options.initialSlew,
                    TransitionEdge::Rise);
    data.addArrival(startNode->getId(), 0, options.initialSlew,
                    TransitionEdge::Fall);
  }

  // Forward propagation in topological order
  for (auto *node : graph.getTopologicalOrder()) {
    auto &nodeData = arrivalData[node->getId().index];
    double outputLoad =
        getNodeOutputLoad(node, delayModel, options.loadSlewHints);

    // For each fanout arc, propagate arrival time
    for (auto *arc : node->getFanout()) {
      auto *successor = arc->getTo();
      auto &succData = arrivalData[successor->getId().index];

      // Get timing sense for this arc
      auto sense = getArcTimingSense(arc, delayModel);

      // Propagate all arrivals from this node
      for (const auto &arrival : nodeData.getAllArrivals()) {
        auto outputEdges = getOutputEdges(arrival.edge, sense);
        for (auto outEdge : outputEdges) {
          auto delay =
              getArcDelay(arc, delayModel, arrival.slew, outputLoad, outEdge);
          int64_t newArrival = arrival.arrivalTime + delay.delay;
          succData.addArrival(arrival.startPoint, newArrival, delay.outputSlew,
                              outEdge);
        }
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

double ArrivalAnalysis::getMaxArrivalSlew(TimingNodeId nodeId) const {
  if (auto *data = getArrivalData(nodeId))
    return data->getMaxArrivalSlew();
  return 0.0;
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
