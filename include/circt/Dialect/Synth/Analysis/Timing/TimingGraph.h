//===- TimingGraph.h - Timing Graph Data Structure -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TimingGraph data structure for static timing analysis.
// The timing graph represents the design as a directed acyclic graph where:
// - Nodes represent timing points (pins, ports, sequential elements)
// - Arcs represent timing relationships with associated delays
//
// This follows the standard EDA timing graph model used in commercial tools.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGGRAPH_H
#define CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGGRAPH_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <memory>
#include <string>
#include <tuple>

namespace circt {
namespace synth {
namespace timing {

class TimingGraph;
class TimingNode;
class TimingArc;

//===----------------------------------------------------------------------===//
// TimingNodeId - Lightweight identifier for timing nodes
//===----------------------------------------------------------------------===//

/// A lightweight identifier for a timing node. Uses an index into the graph's
/// node array for efficient storage and comparison.
struct TimingNodeId {
  uint32_t index = UINT32_MAX;

  bool isValid() const { return index != UINT32_MAX; }
  bool operator==(const TimingNodeId &other) const {
    return index == other.index;
  }
  bool operator!=(const TimingNodeId &other) const { return !(*this == other); }
  bool operator<(const TimingNodeId &other) const {
    return index < other.index;
  }
};

//===----------------------------------------------------------------------===//
// TimingNodeKind - Classification of timing nodes
//===----------------------------------------------------------------------===//

/// Classification of timing node types.
enum class TimingNodeKind : uint8_t {
  InputPort,      // Module input port (start point)
  OutputPort,     // Module output port (end point)
  RegisterOutput, // Register Q output (start point)
  RegisterInput,  // Register D input (end point)
  Combinational,  // Combinational logic node
};

//===----------------------------------------------------------------------===//
// TimingNode - A node in the timing graph
//===----------------------------------------------------------------------===//

/// Represents a timing point in the design. Each node corresponds to a
/// specific bit of a Value in the IR.
class TimingNode {
public:
  TimingNode(TimingNodeId id, Value value, uint32_t bitPos, TimingNodeKind kind,
             StringRef name)
      : id(id), value(value), bitPos(bitPos), kind(kind), name(name.str()) {}

  TimingNodeId getId() const { return id; }
  Value getValue() const { return value; }
  uint32_t getBitPos() const { return bitPos; }
  TimingNodeKind getKind() const { return kind; }
  StringRef getName() const { return name; }

  bool isStartPoint() const {
    return kind == TimingNodeKind::InputPort ||
           kind == TimingNodeKind::RegisterOutput;
  }

  bool isEndPoint() const {
    return kind == TimingNodeKind::OutputPort ||
           kind == TimingNodeKind::RegisterInput;
  }

  // Fanin/fanout access
  ArrayRef<TimingArc *> getFanin() const { return fanin; }
  ArrayRef<TimingArc *> getFanout() const { return fanout; }

  void addFanin(TimingArc *arc) { fanin.push_back(arc); }
  void addFanout(TimingArc *arc) { fanout.push_back(arc); }

private:
  TimingNodeId id;
  Value value;
  uint32_t bitPos;
  TimingNodeKind kind;
  std::string name;

  SmallVector<TimingArc *, 4> fanin;  // Arcs coming into this node
  SmallVector<TimingArc *, 4> fanout; // Arcs going out of this node
};

//===----------------------------------------------------------------------===//
// TimingArc - An edge in the timing graph
//===----------------------------------------------------------------------===//

/// Represents a timing arc (edge) between two timing nodes.
class TimingArc {
public:
  TimingArc(TimingNode *from, TimingNode *to, int64_t delay)
      : from(from), to(to), delay(delay) {}

  TimingNode *getFrom() const { return from; }
  TimingNode *getTo() const { return to; }
  int64_t getDelay() const { return delay; }

private:
  TimingNode *from;
  TimingNode *to;
  int64_t delay;
};

//===----------------------------------------------------------------------===//
// TimingGraph - The main timing graph container
//===----------------------------------------------------------------------===//

/// The timing graph represents the entire design for timing analysis.
/// It owns all nodes and arcs, and provides efficient lookup and traversal.
class TimingGraph {
public:
  explicit TimingGraph(hw::HWModuleOp module);
  TimingGraph(mlir::ModuleOp circuit, hw::HWModuleOp topModule);
  ~TimingGraph();

  // Non-copyable, movable
  TimingGraph(const TimingGraph &) = delete;
  TimingGraph &operator=(const TimingGraph &) = delete;
  TimingGraph(TimingGraph &&) = default;
  TimingGraph &operator=(TimingGraph &&) = default;

  /// Get the module this graph represents.
  hw::HWModuleOp getModule() const { return module; }

  /// Get total number of nodes.
  size_t getNumNodes() const { return nodes.size(); }

  /// Get total number of arcs.
  size_t getNumArcs() const { return arcs.size(); }

  /// Get a node by its ID.
  TimingNode *getNode(TimingNodeId id) const {
    if (id.index >= nodes.size())
      return nullptr;
    return nodes[id.index].get();
  }

  /// Find a node for a specific (Value, bitPos) pair.
  TimingNode *findNode(Value value, uint32_t bitPos) const;

  /// Get all start point nodes (input ports, register outputs).
  ArrayRef<TimingNode *> getStartPoints() const { return startPoints; }

  /// Get all end point nodes (output ports, register inputs).
  ArrayRef<TimingNode *> getEndPoints() const { return endPoints; }

  /// Get all nodes in topological order (for forward traversal).
  ArrayRef<TimingNode *> getTopologicalOrder() const { return topoOrder; }

  /// Get all nodes in reverse topological order (for backward traversal).
  ArrayRef<TimingNode *> getReverseTopologicalOrder() const {
    return reverseTopoOrder;
  }

  /// Get all nodes.
  ArrayRef<std::unique_ptr<TimingNode>> getNodes() const { return nodes; }

  /// Build the timing graph from the module.
  /// If delayModel is null, uses the default AIGLevelDelayModel.
  LogicalResult build(const DelayModel *delayModel = nullptr);

  /// Get the name of the delay model used to build this graph.
  StringRef getDelayModelName() const { return delayModelName; }

private:
  using ValueKey = std::tuple<StringAttr, Value, uint32_t>;

  /// Create a new node and return its ID.
  TimingNodeId createNode(Value value, uint32_t bitPos, TimingNodeKind kind,
                          StringRef name, StringRef contextPath,
                          bool addToLookup = true);

  /// Create a new arc between two nodes.
  TimingArc *createArc(TimingNode *from, TimingNode *to, int64_t delay);

  /// Compute topological ordering of nodes.
  void computeTopologicalOrder();

  /// Get or create a node for a value/bit.
  TimingNode *getOrCreateNode(Value value, uint32_t bitPos,
                              hw::HWModuleOp currentModule,
                              StringRef contextPath, bool topContext);

  TimingNode *findNode(Value value, uint32_t bitPos,
                       StringRef contextPath) const;

  /// Process an operation to create nodes and arcs.
  LogicalResult processOperation(Operation *op, const DelayModel &model,
                                 hw::HWModuleOp currentModule,
                                 StringRef contextPath, bool topContext);

  LogicalResult buildFlatGraph(const DelayModel &model);
  LogicalResult buildHierarchicalGraph(const DelayModel &model);
  LogicalResult buildModuleInContext(const DelayModel &model,
                                     hw::HWModuleOp currentModule,
                                     StringRef contextPath,
                                     llvm::SmallVectorImpl<StringAttr> &stack,
                                     bool topContext);

  /// Get the name for a value.
  std::string getNameForValue(Value value, hw::HWModuleOp currentModule,
                              StringRef contextPath) const;

  mlir::ModuleOp circuit;
  hw::HWModuleOp module;
  bool hierarchical = false;
  std::string delayModelName;

  // Keep nodes in std::unique_ptr because TimingNode owns non-trivial members
  // (SmallVector fanin/fanout and std::string name). Arcs are much lighter and
  // far more numerous, so they use a bump allocator for lower allocation
  // overhead while preserving graph-lifetime ownership semantics.
  SmallVector<std::unique_ptr<TimingNode>> nodes;
  SmallVector<TimingArc *> arcs;
  llvm::SpecificBumpPtrAllocator<TimingArc> arcAllocator;
  SmallVector<TimingNode *> startPoints;
  SmallVector<TimingNode *> endPoints;
  SmallVector<TimingNode *> topoOrder;
  SmallVector<TimingNode *> reverseTopoOrder;

  // Lookup map: (Value, bitPos) -> TimingNode*
  DenseMap<ValueKey, TimingNode *> valueToNode;
};

} // namespace timing
} // namespace synth
} // namespace circt

// DenseMapInfo for TimingNodeId
namespace llvm {
template <>
struct DenseMapInfo<circt::synth::timing::TimingNodeId> {
  static circt::synth::timing::TimingNodeId getEmptyKey() {
    return {UINT32_MAX};
  }
  static circt::synth::timing::TimingNodeId getTombstoneKey() {
    return {UINT32_MAX - 1};
  }
  static unsigned getHashValue(circt::synth::timing::TimingNodeId id) {
    return DenseMapInfo<uint32_t>::getHashValue(id.index);
  }
  static bool isEqual(circt::synth::timing::TimingNodeId lhs,
                      circt::synth::timing::TimingNodeId rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

#endif // CIRCT_DIALECT_SYNTH_ANALYSIS_TIMING_TIMINGGRAPH_H
