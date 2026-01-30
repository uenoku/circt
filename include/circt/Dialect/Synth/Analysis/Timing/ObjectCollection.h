//===- ObjectCollection.h - Object Matching and Collections ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for matching and collecting design objects
// using glob patterns. This provides functionality similar to EDA tool
// commands like get_cells, get_ports, get_pins.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_OBJECTCOLLECTION_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_OBJECTCOLLECTION_H

#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/GlobPattern.h"

namespace circt {
namespace synth {
namespace timing {

//===----------------------------------------------------------------------===//
// ObjectMatcher - Pattern matching for timing nodes
//===----------------------------------------------------------------------===//

/// Matches timing nodes against glob patterns.
class ObjectMatcher {
public:
  /// Create a matcher from a list of glob patterns.
  static llvm::Expected<ObjectMatcher>
  create(ArrayRef<std::string> patterns);

  /// Check if a node name matches any pattern.
  bool matches(StringRef name) const;

  /// Check if any patterns are specified.
  bool hasPatterns() const { return !patterns.empty(); }

  /// Check if this is a passthrough matcher (no patterns = match all).
  bool isPassthrough() const { return patterns.empty(); }

private:
  SmallVector<llvm::GlobPattern, 2> patterns;
};

//===----------------------------------------------------------------------===//
// ObjectCollection - A collection of timing nodes
//===----------------------------------------------------------------------===//

/// Represents a collection of timing nodes, similar to EDA tool collections.
class ObjectCollection {
public:
  ObjectCollection() = default;

  /// Create a collection from explicit nodes.
  explicit ObjectCollection(ArrayRef<TimingNode *> nodes)
      : nodes(nodes.begin(), nodes.end()) {}

  /// Create a collection by matching patterns against the graph.
  static ObjectCollection fromPatterns(const TimingGraph &graph,
                                       ArrayRef<std::string> patterns);

  /// Create a collection of all start points.
  static ObjectCollection allStartPoints(const TimingGraph &graph);

  /// Create a collection of all end points.
  static ObjectCollection allEndPoints(const TimingGraph &graph);

  /// Create a collection of all nodes (including intermediate).
  static ObjectCollection allNodes(const TimingGraph &graph);

  /// Get the nodes in this collection.
  ArrayRef<TimingNode *> getNodes() const { return nodes; }

  /// Get the number of nodes.
  size_t size() const { return nodes.size(); }

  /// Check if empty.
  bool empty() const { return nodes.empty(); }

  /// Filter this collection to only start points.
  ObjectCollection filterStartPoints() const;

  /// Filter this collection to only end points.
  ObjectCollection filterEndPoints() const;

  /// Filter this collection to only nodes of a specific kind.
  ObjectCollection filterByKind(TimingNodeKind kind) const;

  /// Intersect with another collection.
  ObjectCollection intersect(const ObjectCollection &other) const;

  /// Union with another collection.
  ObjectCollection unite(const ObjectCollection &other) const;

  /// Get the node names.
  SmallVector<std::string> getNames() const;

private:
  SmallVector<TimingNode *, 8> nodes;
};

//===----------------------------------------------------------------------===//
// Convenience functions
//===----------------------------------------------------------------------===//

/// Match nodes in the graph against patterns, returning matched nodes.
SmallVector<TimingNode *> matchNodes(const TimingGraph &graph,
                                     ArrayRef<std::string> patterns);

/// Match start points against patterns.
SmallVector<TimingNode *> matchStartPoints(const TimingGraph &graph,
                                           ArrayRef<std::string> patterns);

/// Match end points against patterns.
SmallVector<TimingNode *> matchEndPoints(const TimingGraph &graph,
                                         ArrayRef<std::string> patterns);

} // namespace timing
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_OBJECTCOLLECTION_H

