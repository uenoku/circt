//===- PathEnumerator.cpp - Path Enumeration Engine -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/PathEnumerator.h"
#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <queue>

#define DEBUG_TYPE "path-enumerator"

using namespace circt;
using namespace circt::synth::timing;

static int64_t getArcDelay(const TimingArc *arc, const DelayModel *delayModel) {
  if (!delayModel || !arc->getOp())
    return arc->getDelay();

  DelayContext ctx;
  ctx.op = arc->getOp();
  ctx.inputValue = arc->getInputValue();
  ctx.outputValue = arc->getOutputValue();
  ctx.inputIndex = arc->getInputIndex();
  ctx.outputIndex = arc->getOutputIndex();
  return delayModel->computeDelay(ctx).delay;
}

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
                               const ArrivalAnalysis &arrivals,
                               const DelayModel *delayModel)
    : graph(graph), arrivals(arrivals), delayModel(delayModel) {}

void PathEnumerator::buildSuffixTree(
    TimingNode *endpoint, llvm::DenseMap<TimingNode *, SuffixTreeEntry> &sfxt) {
  sfxt.clear();

  // Initialize endpoint
  sfxt[endpoint] = {0, nullptr, nullptr};

  // Backward DP in reverse topological order
  // We want longest path to endpoint (for worst-case timing),
  // so dist = max distance to endpoint.
  for (auto *node : graph.getReverseTopologicalOrder()) {
    for (auto *arc : node->getFanout()) {
      auto *succ = arc->getTo();
      auto succIt = sfxt.find(succ);
      if (succIt == sfxt.end())
        continue;

      int64_t candidateDist =
          succIt->second.dist + getArcDelay(arc, delayModel);
      auto &entry = sfxt[node];
      if (entry.dist < candidateDist) {
        entry.dist = candidateDist;
        entry.parent = succ;
        entry.arc = arc;
      }
    }
  }
}

void PathEnumerator::extractKPaths(
    TimingNode *endpoint, size_t k,
    const llvm::DenseMap<TimingNode *, SuffixTreeEntry> &sfxt,
    const llvm::DenseSet<TimingNode *> &validStartPoints,
    SmallVectorImpl<TimingPath> &results) {

  // Collect all reachable start points and their distances
  struct StartEntry {
    TimingNode *node;
    int64_t dist;
    bool operator<(const StartEntry &o) const { return dist < o.dist; }
  };

  // Use a max-heap: pop the longest-distance start points first
  std::priority_queue<StartEntry> pq;

  for (const auto &[node, entry] : sfxt) {
    if (!node->isStartPoint())
      continue;
    if (!validStartPoints.empty() && !validStartPoints.contains(node))
      continue;
    pq.push({node, entry.dist});
  }

  size_t added = 0;
  while (!pq.empty() && (k == 0 || added < k)) {
    auto [startNode, dist] = pq.top();
    pq.pop();
    results.emplace_back(startNode, endpoint, dist);
    ++added;
  }
}

void PathEnumerator::reconstructPathViaSFXT(
    TimingNode *start, TimingNode *end,
    const llvm::DenseMap<TimingNode *, SuffixTreeEntry> &sfxt,
    TimingPath &path) {
  SmallVector<TimingNode *, 16> intermediate;

  auto *cur = start;
  while (cur != end) {
    auto it = sfxt.find(cur);
    if (it == sfxt.end() || !it->second.parent)
      break;
    cur = it->second.parent;
    if (cur != end)
      intermediate.push_back(cur);
  }
  path.setIntermediateNodes(intermediate);
}

bool PathEnumerator::goesThrough(
    const TimingPath &path, const llvm::DenseSet<TimingNode *> &throughNodes) {
  if (throughNodes.empty())
    return true;
  // Check start and end
  if (throughNodes.contains(path.getStartPoint()) ||
      throughNodes.contains(path.getEndPoint()))
    return true;
  // Check intermediate nodes (requires reconstruction)
  for (auto *node : path.getIntermediateNodes()) {
    if (throughNodes.contains(node))
      return true;
  }
  return false;
}

LogicalResult PathEnumerator::enumerate(const PathQuery &query,
                                        SmallVectorImpl<TimingPath> &results) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating paths with query\n");

  // Match start points
  SmallVector<TimingNode *> fromNodes;
  if (query.fromPatterns.empty()) {
    fromNodes.assign(arrivals.getMatchedStartPoints().begin(),
                     arrivals.getMatchedStartPoints().end());
  } else {
    fromNodes = matchStartPoints(graph, query.fromPatterns);
  }

  // Match end points
  SmallVector<TimingNode *> toNodes;
  if (query.toPatterns.empty()) {
    toNodes.assign(graph.getEndPoints().begin(), graph.getEndPoints().end());
  } else {
    toNodes = timing::matchNodes(graph, query.toPatterns);
  }

  // Match through points if specified
  llvm::DenseSet<TimingNode *> throughNodes;
  if (!query.throughPatterns.empty()) {
    auto matched = timing::matchNodes(graph, query.throughPatterns);
    throughNodes.insert(matched.begin(), matched.end());
  }

  llvm::DenseSet<TimingNode *> validStartPoints(fromNodes.begin(),
                                                fromNodes.end());

  LLVM_DEBUG(llvm::dbgs() << "From nodes: " << fromNodes.size()
                          << ", To nodes: " << toNodes.size() << "\n");

  // Strategy depends on whether through-point filtering is needed:
  //
  // With throughNodes: reconstruct eagerly (before filter) so that
  //   goesThrough() can inspect intermediate nodes. The sfxt is consumed
  //   per-endpoint and can be freed immediately.
  //
  // Without throughNodes (common case): keep all sfxts alive, do global
  //   sort+trim first, then reconstruct only the surviving paths. This avoids
  //   paying O(depth) reconstruction for paths that get discarded.

  mlir::MLIRContext *ctx = graph.getModule()->getContext();
  const size_t numEndpoints = toNodes.size();

  if (!throughNodes.empty()) {
    // Eager path: reconstruct + filter per endpoint, then merge.
    // Each endpoint slot is independent so we can build sfxt and collect
    // candidates in parallel into per-endpoint local vectors.
    using SfxtMap = llvm::DenseMap<TimingNode *, SuffixTreeEntry>;
    SmallVector<SfxtMap> sfxts(numEndpoints);
    SmallVector<SmallVector<TimingPath>> localResults(numEndpoints);

    mlir::parallelFor(ctx, 0, numEndpoints, [&](size_t i) {
      auto *endpoint = toNodes[i];
      buildSuffixTree(endpoint, sfxts[i]);
      extractKPaths(endpoint, query.maxPaths, sfxts[i], validStartPoints,
                    localResults[i]);
      // Reconstruct and filter against throughNodes.
      auto &local = localResults[i];
      for (size_t j = 0; j < local.size();) {
        reconstructPathViaSFXT(local[j].getStartPoint(), endpoint, sfxts[i],
                               local[j]);
        if (!goesThrough(local[j], throughNodes)) {
          local[j] = std::move(local.back());
          local.pop_back();
        } else {
          ++j;
        }
      }
    });

    // Merge per-endpoint results.
    for (auto &local : localResults)
      for (auto &p : local)
        results.push_back(std::move(p));

    llvm::sort(results, [](const TimingPath &a, const TimingPath &b) {
      return a.getDelay() > b.getDelay();
    });
    if (query.maxPaths > 0 && results.size() > query.maxPaths)
      results.resize(query.maxPaths);

  } else {
    // Deferred path: build sfxts and collect candidates in parallel, then
    // sort+trim sequentially, then reconstruct only the survivors in parallel.
    using SfxtMap = llvm::DenseMap<TimingNode *, SuffixTreeEntry>;
    SmallVector<SfxtMap> sfxts(numEndpoints);
    SmallVector<SmallVector<TimingPath>> localResults(numEndpoints);

    mlir::parallelFor(ctx, 0, numEndpoints, [&](size_t i) {
      buildSuffixTree(toNodes[i], sfxts[i]);
      extractKPaths(toNodes[i], query.maxPaths, sfxts[i], validStartPoints,
                    localResults[i]);
    });

    // Merge per-endpoint results.
    for (auto &local : localResults)
      for (auto &p : local)
        results.push_back(std::move(p));

    llvm::sort(results, [](const TimingPath &a, const TimingPath &b) {
      return a.getDelay() > b.getDelay();
    });
    if (query.maxPaths > 0 && results.size() > query.maxPaths)
      results.resize(query.maxPaths);

    if (query.reconstructPaths) {
      // Build endpoint -> sfxt index map for O(1) lookup.
      llvm::DenseMap<TimingNode *, unsigned> endpointToSfxt;
      for (auto [i, endpoint] : llvm::enumerate(toNodes))
        endpointToSfxt[endpoint] = i;

      // Reconstruct surviving paths in parallel — each path is independent.
      mlir::parallelFor(ctx, 0, results.size(), [&](size_t i) {
        auto &path = results[i];
        auto it = endpointToSfxt.find(path.getEndPoint());
        if (it == endpointToSfxt.end())
          return;
        reconstructPathViaSFXT(path.getStartPoint(), path.getEndPoint(),
                               sfxts[it->second], path);
      });
    }
  }

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
PathEnumerator::getKWorstPaths(size_t k, SmallVectorImpl<TimingPath> &results) {
  PathQuery query;
  query.maxPaths = k;
  return enumerate(query, results);
}
