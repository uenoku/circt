//===- PathEnumerator.cpp - Path Enumeration Engine -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/PathEnumerator.h"
#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "path-enumerator"

using namespace circt;
using namespace circt::synth::timing;

//===----------------------------------------------------------------------===//
// TimingPath Implementation
//===----------------------------------------------------------------------===//

void TimingPath::print(llvm::raw_ostream &os) const {
  os << "[delay=" << delay << "] " << startPoint->getName() << " -> "
     << endPoint->getName();
}

//===----------------------------------------------------------------------===//
// PathEnumerator Implementation
//===----------------------------------------------------------------------===//

PathEnumerator::PathEnumerator(const TimingGraph &graph,
                               const ArrivalAnalysis &arrivals)
    : graph(graph), arrivals(arrivals) {}

LogicalResult PathEnumerator::enumerate(const PathQuery &query,
                                        SmallVectorImpl<TimingPath> &results) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating paths with query\n");

  // Match start points
  SmallVector<TimingNode *> fromNodes;
  if (query.fromPatterns.empty()) {
    // Use matched start points from arrival analysis
    fromNodes.assign(arrivals.getMatchedStartPoints().begin(),
                     arrivals.getMatchedStartPoints().end());
  } else {
    fromNodes = matchStartPoints(graph, query.fromPatterns);
  }

  // Match end points
  SmallVector<TimingNode *> toNodes;
  if (query.toPatterns.empty()) {
    if (query.includeIntermediateEndpoints) {
      // Include all nodes with arrivals
      for (const auto &node : graph.getNodes()) {
        if (arrivals.getArrivalData(node.get())->hasArrivals())
          toNodes.push_back(node.get());
      }
    } else {
      // Only formal end points
      toNodes.assign(graph.getEndPoints().begin(), graph.getEndPoints().end());
    }
  } else {
    toNodes = timing::matchNodes(graph, query.toPatterns);
  }

  // Match through points if specified
  SmallVector<TimingNode *> throughNodes;
  if (!query.throughPatterns.empty())
    throughNodes = timing::matchNodes(graph, query.throughPatterns);

  LLVM_DEBUG(llvm::dbgs() << "From nodes: " << fromNodes.size()
                          << ", To nodes: " << toNodes.size()
                          << ", Through nodes: " << throughNodes.size() << "\n");

  // Create a set of valid start points for filtering
  llvm::DenseSet<TimingNodeId> validStartPoints;
  for (auto *node : fromNodes)
    validStartPoints.insert(node->getId());

  // Enumerate paths
  for (auto *endNode : toNodes) {
    auto *arrivalData = arrivals.getArrivalData(endNode);
    if (!arrivalData || !arrivalData->hasArrivals())
      continue;

    // Get arrivals at this end point
    for (const auto &arrival : arrivalData->getAllArrivals()) {
      // Check if this arrival is from a valid start point
      if (!validStartPoints.contains(arrival.startPoint))
        continue;

      auto *startNode = graph.getNode(arrival.startPoint);
      if (!startNode)
        continue;

      TimingPath path(startNode, endNode, arrival.arrivalTime);

      // Check through constraint if specified
      if (!throughNodes.empty() && !goesThrough(path, throughNodes))
        continue;

      // Reconstruct intermediate nodes if requested
      if (query.reconstructPaths)
        reconstructPath(path);

      results.push_back(std::move(path));

      // Check max paths limit
      if (query.maxPaths > 0 && results.size() >= query.maxPaths)
        break;
    }

    if (query.maxPaths > 0 && results.size() >= query.maxPaths)
      break;
  }

  // Sort by delay (descending)
  std::sort(results.begin(), results.end(),
            [](const TimingPath &a, const TimingPath &b) {
              return a.getDelay() > b.getDelay();
            });

  // Trim to maxPaths if needed
  if (query.maxPaths > 0 && results.size() > query.maxPaths)
    results.resize(query.maxPaths);

  LLVM_DEBUG(llvm::dbgs() << "Enumerated " << results.size() << " paths\n");

  return success();
}

LogicalResult
PathEnumerator::getPathsToNode(TimingNode *node,
                               SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.toPatterns = {node->getName().str()};
  return enumerate(query, results);
}

LogicalResult
PathEnumerator::getPathsFromNode(TimingNode *startPoint,
                                 SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.fromPatterns = {startPoint->getName().str()};
  return enumerate(query, results);
}

LogicalResult
PathEnumerator::getKWorstPaths(size_t k,
                               SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.maxPaths = k;
  return enumerate(query, results);
}

void PathEnumerator::reconstructPath(TimingPath &path) {
  // TODO: Implement backward trace from end to start
  // For now, leave intermediate nodes empty
}

bool PathEnumerator::goesThrough(const TimingPath &path,
                                 ArrayRef<TimingNode *> throughNodes) {
  // TODO: Implement through-point checking
  // Would require path reconstruction
  return true;
}

