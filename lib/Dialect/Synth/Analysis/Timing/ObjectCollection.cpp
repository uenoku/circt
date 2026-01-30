//===- ObjectCollection.cpp - Object Matching and Collections ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/ObjectCollection.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "object-collection"

using namespace circt;
using namespace circt::synth::timing;

//===----------------------------------------------------------------------===//
// ObjectMatcher Implementation
//===----------------------------------------------------------------------===//

llvm::Expected<ObjectMatcher>
ObjectMatcher::create(ArrayRef<std::string> patternStrs) {
  ObjectMatcher matcher;

  for (const auto &patStr : patternStrs) {
    if (patStr.empty())
      continue;
    auto pat = llvm::GlobPattern::create(patStr);
    if (!pat)
      return pat.takeError();
    matcher.patterns.push_back(std::move(*pat));
  }

  return matcher;
}

bool ObjectMatcher::matches(StringRef name) const {
  if (patterns.empty())
    return true; // No patterns = match all

  for (const auto &pat : patterns) {
    if (pat.match(name))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ObjectCollection Implementation
//===----------------------------------------------------------------------===//

ObjectCollection ObjectCollection::fromPatterns(const TimingGraph &graph,
                                                ArrayRef<std::string> patterns) {
  if (patterns.empty())
    return allNodes(graph);

  auto matcherOrErr = ObjectMatcher::create(patterns);
  if (!matcherOrErr) {
    llvm::consumeError(matcherOrErr.takeError());
    return ObjectCollection();
  }

  auto &matcher = *matcherOrErr;
  SmallVector<TimingNode *, 8> matched;

  for (const auto &node : graph.getNodes()) {
    if (matcher.matches(node->getName()))
      matched.push_back(node.get());
  }

  return ObjectCollection(matched);
}

ObjectCollection ObjectCollection::allStartPoints(const TimingGraph &graph) {
  return ObjectCollection(graph.getStartPoints());
}

ObjectCollection ObjectCollection::allEndPoints(const TimingGraph &graph) {
  return ObjectCollection(graph.getEndPoints());
}

ObjectCollection ObjectCollection::allNodes(const TimingGraph &graph) {
  SmallVector<TimingNode *, 64> allNodes;
  for (const auto &node : graph.getNodes())
    allNodes.push_back(node.get());
  return ObjectCollection(allNodes);
}

ObjectCollection ObjectCollection::filterStartPoints() const {
  SmallVector<TimingNode *, 8> filtered;
  for (auto *node : nodes) {
    if (node->isStartPoint())
      filtered.push_back(node);
  }
  return ObjectCollection(filtered);
}

ObjectCollection ObjectCollection::filterEndPoints() const {
  SmallVector<TimingNode *, 8> filtered;
  for (auto *node : nodes) {
    if (node->isEndPoint())
      filtered.push_back(node);
  }
  return ObjectCollection(filtered);
}

ObjectCollection ObjectCollection::filterByKind(TimingNodeKind kind) const {
  SmallVector<TimingNode *, 8> filtered;
  for (auto *node : nodes) {
    if (node->getKind() == kind)
      filtered.push_back(node);
  }
  return ObjectCollection(filtered);
}

ObjectCollection
ObjectCollection::intersect(const ObjectCollection &other) const {
  llvm::DenseSet<TimingNode *> otherSet(other.nodes.begin(), other.nodes.end());
  SmallVector<TimingNode *, 8> result;
  for (auto *node : nodes) {
    if (otherSet.contains(node))
      result.push_back(node);
  }
  return ObjectCollection(result);
}

ObjectCollection ObjectCollection::unite(const ObjectCollection &other) const {
  llvm::DenseSet<TimingNode *> seen(nodes.begin(), nodes.end());
  SmallVector<TimingNode *, 8> result(nodes.begin(), nodes.end());
  for (auto *node : other.nodes) {
    if (seen.insert(node).second)
      result.push_back(node);
  }
  return ObjectCollection(result);
}

SmallVector<std::string> ObjectCollection::getNames() const {
  SmallVector<std::string> names;
  names.reserve(nodes.size());
  for (auto *node : nodes)
    names.push_back(node->getName().str());
  return names;
}

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

SmallVector<TimingNode *>
circt::synth::timing::matchNodes(const TimingGraph &graph,
                                 ArrayRef<std::string> patterns) {
  auto nodes = ObjectCollection::fromPatterns(graph, patterns).getNodes();
  return SmallVector<TimingNode *>(nodes.begin(), nodes.end());
}

SmallVector<TimingNode *>
circt::synth::timing::matchStartPoints(const TimingGraph &graph,
                                       ArrayRef<std::string> patterns) {
  auto nodes = ObjectCollection::fromPatterns(graph, patterns)
                   .filterStartPoints()
                   .getNodes();
  return SmallVector<TimingNode *>(nodes.begin(), nodes.end());
}

SmallVector<TimingNode *>
circt::synth::timing::matchEndPoints(const TimingGraph &graph,
                                     ArrayRef<std::string> patterns) {
  auto nodes = ObjectCollection::fromPatterns(graph, patterns)
                   .filterEndPoints()
                   .getNodes();
  return SmallVector<TimingNode *>(nodes.begin(), nodes.end());
}

