//===- ResourceUsageAnalysis.cpp - resource usage analysis ---------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the critical path analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"

#include "llvm/Support/raw_ostream.h"

#include <queue>

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

static size_t getBitWidth(Value value) {
  if (auto vecType = value.getType().dyn_cast<seq::ClockType>())
    return 1;
  return hw::getBitWidth(value.getType());
}

static bool isRootValue(Value value) {
  if (auto arg = value.dyn_cast<BlockArgument>())
    return true;
  return isa<seq::CompRegOp, seq::FirRegOp, hw::InstanceOp, seq::FirMemReadOp>(
      value.getDefiningOp());
}

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#define GEN_PASS_DEF_PRINTGLOBALDATAPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

struct InputNode;
struct Graph;

struct Node {
  enum Kind {
    Input,
    Delay,
    Output,
    Concat,
    Replicate,
    Extract,
    Constant,
    Debug,
  };
  Kind kind;
  size_t width;
  Graph *graph;
  circt::igraph::InstancePath path;
  void setPath(circt::igraph::InstancePath path) { this->path = path; }
  Node(Graph *graph, Kind kind, size_t width)
      : graph(graph), kind(kind), width(width), path() {}

  size_t getWidth() const { return width; }
  Kind getKind() const { return kind; }
  Graph *getGraph() const { return graph; }

  virtual void
  populateResults(SmallVector<std::pair<int64_t, InputNode *>> &results) {};

  virtual Node *query(size_t bitOffset) = 0;
  virtual ~Node() {}
  virtual void dump(size_t indent = 0) const = 0;
  virtual void print(llvm::raw_ostream &os) const { assert(false); };
  virtual void walk(llvm::function_ref<void(Node *)>) = 0;
  virtual void walkPreOrder(llvm::function_ref<bool(Node *)>){};
  void preorderOnce(llvm::function_ref<bool(Node *)> func) {
    DenseSet<Node *> visited;
    auto callback = [&](Node *node) {
      if (!visited.insert(node).second)
        return false;
      return func(node);
    };
    walk(callback);
  }

  virtual void map(DenseMap<InputNode *, Node *> &nodeToNewNode) {
    assert(false);
  };
};

struct DebugNode : Node {
  DebugNode(Graph *graph, StringAttr name, size_t bitPos, Node *input)
      : Node(graph, Kind::Debug, 1), name(name), bitPos(bitPos), input(input) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Debug; }
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent)
        << "(debug node " << name << " " << bitPos << ")";
  }
  void print(llvm::raw_ostream &os) const override {
    std::string pathString;
    llvm::raw_string_ostream osPath(pathString);
    path.print(osPath);
    os << "DebugNode(" << pathString << "." << name << "[" << bitPos << "])";
  }

  Node *query(size_t bitOffset) override { return input->query(bitOffset); }

private:
  StringAttr name;
  size_t bitPos;
  Node *input;
};

struct ConstantNode : Node {
  ConstantNode(Graph *graph, size_t width, ConstantNode *boolNode)
      : Node(graph, Kind::Constant, width), boolNode(boolNode) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Constant; }
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent) << "(constant node " << width << ")";
  }
  Node *query(size_t bitOffset) override { return boolNode; }

  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  void map(DenseMap<InputNode *, Node *> &nodeToNewNode) override {}

private:
  ConstantNode *boolNode = nullptr;
};

struct InputNode : Node {
  Value value; // port, instance output, or register.
  size_t bitPos;
  InputNode(Graph *graph, Value value, size_t bitPos)
      : Node(graph, Kind::Input, 1), value(value), bitPos(bitPos) {}

  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  static bool classof(const Node *e) { return e->getKind() == Kind::Input; }
  void setBitPos(size_t bitPos) { this->bitPos = bitPos; }
  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "input node has no bit offset");
    return this;
  }
  size_t getBitPos() const { return bitPos; }
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent)
        << "(input node " << value << " " << bitPos << "@" << this << ")";
  }

  StringRef getName() const {
    if (auto arg = value.dyn_cast<BlockArgument>()) {
      auto op = cast<hw::HWModuleOp>(arg.getParentBlock()->getParentOp());
      return op.getArgName(arg.getArgNumber());
    }
    return TypeSwitch<Operation *, StringRef>(value.getDefiningOp())
        .Case<seq::CompRegOp, seq::FirRegOp>(
            [](auto op) { return op.getNameAttr().getValue(); })
        .Case<hw::InstanceOp>([&](hw::InstanceOp op) {
          return cast<StringAttr>(
              op.getResultNamesAttr()[cast<OpResult>(value).getResultNumber()]);
        })
        .Case<seq::FirMemReadOp>([&](seq::FirMemReadOp op) {
          llvm::SmallString<16> str;
          str += op.getMemory().getDefiningOp<seq::FirMemOp>().getNameAttr();
          str += "_read_port";
          return StringAttr::get(value.getContext(), str);
        })
        .Default([](auto op) {
          llvm::errs() << "Unknown op: " << *op << "\n";
          return "";
        });
  }

  llvm::json::Object getJSONObject() {
    llvm::json::Object result;
    std::string pathString;
    llvm::raw_string_ostream os(pathString);
    path.print(os);
    result["hierarchy"] = std::move(pathString);
    result["name"] = getName();
    result["bitPosition"] = bitPos;
    return result;
  }

  void print(llvm::raw_ostream &os) const override {
    std::string pathString;
    llvm::raw_string_ostream osPath(pathString);
    path.print(osPath);
    os << "InputNode(" << pathString << "." << getName() << "[" << bitPos
       << "])";
  }

  void map(DenseMap<InputNode *, Node *> &nodeToNewNode) override {}

  void populateResults(SmallVector<std::pair<int64_t, InputNode *>> &results) {
    results.push_back(std::make_pair(0, this));
  }
};

struct DelayNode : Node {

  DelayNode &operator=(const DelayNode &other) = default;
  DelayNode(const DelayNode &other) = default;
  DelayNode(Graph *graph, Value value, size_t bitPos)
      : Node(graph, Kind::Delay, 1), value(value), bitPos(bitPos), edges(),
        computedResult() {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Delay; }
  SmallVector<std::pair<int64_t, Node *>> getEdges() const { return edges; }
  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  Value getValue() const { return value; }
  size_t getBitPos() const { return bitPos; }

  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "delay node has no bit offset");
    return this;
  }
  void addEdge(Node *node, int64_t delay) {
    revertComputedResult();
    edges.push_back(std::make_pair(delay, node));
  }
  void setBitPos(size_t bitPos) { this->bitPos = bitPos; }

  void map(DenseMap<InputNode *, Node *> &nodeToNewNode) override {
    revertComputedResult();
    // llvm::dbgs() << "Mapping delay node " << edges.size() << "\n";
    for (size_t i = 0, e = edges.size(); i < e; ++i) {
      if (auto *newNode = dyn_cast<InputNode>(edges[i].second)) {
        if (auto *newEdgeNode = nodeToNewNode[newNode]) {
          // llvm::dbgs() << "Updating edge " << i << " from " <<
          // edges[i].second
          //              << " to " << newEdgeNode << "\n";
          edges[i].second = newEdgeNode;
        }
      }
    }
    // llvm::dbgs() << "Done mapping delay node " << edges.size() << "\n";
  }

  void populateResults(
      SmallVector<std::pair<int64_t, InputNode *>> &results) override {
    auto &computedResult = getComputedResult();
    results.append(computedResult.begin(), computedResult.end());
  }
  void revertComputedResult() { computedResult.reset(); }

  void shrinkEdges() {
    // This removes intermediate nodes in the graph.
    SmallVector<std::pair<int64_t, InputNode *>> maxDelays;
    populateResults(maxDelays);
    edges.clear();
    for (auto [delay, inputNode] : maxDelays)
      addEdge(inputNode, delay);
    computedResult = std::move(maxDelays);
  }

  void computeResult() {
    computedResult =
        std::make_optional(SmallVector<std::pair<int64_t, InputNode *>>());
    llvm::MapVector<InputNode *, int64_t> maxDelays;
    for (auto [delay, node] : edges) {
      if (auto *inputNode = dyn_cast<InputNode>(node))
        maxDelays[inputNode] = std::max(maxDelays[inputNode], delay);
      else if (auto *delayNode = dyn_cast<DelayNode>(node)) {
        for (auto [newDelay, inputNode] : delayNode->getComputedResult())
          maxDelays[inputNode] =
              std::max(maxDelays[inputNode], delay + newDelay);
      } else if (auto *concatNode = dyn_cast<ConstantNode>(node)) {
        continue;
      } else
        llvm_unreachable("unknown node type");
    }

    for (auto [inputNode, delay] : maxDelays)
      computedResult->push_back(std::make_pair(delay, inputNode));
  }

  const SmallVector<std::pair<int64_t, InputNode *>> &getComputedResult() {
    if (!computedResult) {
      computeResult();
      shrinkEdges();
    }

    assert(computedResult && "result not computed");
    return computedResult.value();
  }

  std::optional<SmallVector<std::pair<int64_t, InputNode *>>> computedResult;

  ~DelayNode() override {
    edges.clear();
    computedResult.reset();
  }

  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent)
        << "(delay node " << value << " " << bitPos << "\n";
    for (auto edge : edges) {
      llvm::dbgs().indent(indent + 2) << "edge: delay " << edge.first << "\n";
      edge.second->dump(indent + 2);
      llvm::dbgs() << "\n";
    }
    llvm::dbgs().indent(indent) << ")\n";
  }

private:
  Value value;
  size_t bitPos;
  SmallVector<std::pair<int64_t, Node *>> edges;
};

struct OutputNode : Node {
  Operation *op;
  size_t operandIdx;
  size_t bitPos;
  Node *node;

  std::optional<SmallVector<std::pair<int64_t, InputNode *>>> computedResult;

  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  void map(DenseMap<InputNode *, Node *> &nodeToNewNode) override {
    computedResult.reset();
    if (auto *inputNode = dyn_cast<InputNode>(node))
      if (auto *newNode = nodeToNewNode[inputNode])
        node = newNode;
  }

  OutputNode(Graph *graph, Operation *op, size_t operandIdx, size_t bitPos,
             Node *node)
      : Node(graph, Kind::Output, node->getWidth()), op(op),
        operandIdx(operandIdx), bitPos(bitPos), node(node) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Output; }
  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "delay node has no bit offset");
    return node->query(0);
  }

  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent)
        << "(output node " << *op << " " << operandIdx << " " << bitPos << "\n";
    node->dump(indent + 2);
    llvm::dbgs().indent(indent) << ")";
  }

  void print(llvm::raw_ostream &os) const override {
    std::string pathString;
    llvm::raw_string_ostream osPath(pathString);
    path.print(osPath);
    os << "OutputNode(" << pathString << "." << getName() << "[" << bitPos
       << "])";
  }

  StringRef getName() const {
    return TypeSwitch<Operation *, StringRef>(op)
        .Case<seq::CompRegOp, seq::FirRegOp>([&](auto op) {
          auto ref = op.getNameAttr().getValue();
          if (ref.empty())
            return StringRef("<anonymous_reg>");
          return ref;
        })
        .Case<hw::OutputOp>([&](hw::OutputOp op) {
          auto hwModule = cast<hw::HWModuleOp>(op.getParentOp());
          return hwModule.getOutputNameAttr(operandIdx).getValue();
        })
        .Default([&](auto op) {
          llvm::dbgs() << "Unknown op: " << *op << "\n";
          return "";
        });
  }

  void populateResults(
      SmallVector<std::pair<int64_t, InputNode *>> &results) override {
    node->populateResults(results);
  }

  const SmallVector<std::pair<int64_t, InputNode *>> &getComputedResult() {
    if (!computedResult) {
      // TODO: compute result
      computedResult =
          std::make_optional(SmallVector<std::pair<int64_t, InputNode *>>());
      populateResults(computedResult.value());
    }

    assert(computedResult && "result not computed");
    return computedResult.value();
  }

  llvm::json::Object getJSONObject() {
    llvm::json::Object result;
    std::string pathString;
    llvm::raw_string_ostream os(pathString);
    path.print(os);
    result["hierarchy"] = std::move(pathString);
    result["name"] = getName();
    result["bitPosition"] = bitPos;
    SmallVector<llvm::json::Value> fanIn;
    for (auto [delay, inputNode] : getComputedResult()) {
      llvm::json::Object fanInObj;
      fanInObj["delay"] = delay;
      fanInObj["node"] = inputNode->getJSONObject();
      fanIn.push_back(std::move(fanInObj));
    }

    result["fanIns"] = llvm::json::Array(std::move(fanIn));
    return result;
  }

  ~OutputNode() = default;
};

struct ConcatNode : Node {
  Value value;
  SmallVector<Node *> nodes;
  ConcatNode(Graph *graph, Value value, ArrayRef<Node *> nodes)
      : Node(graph, Kind::Concat, getBitWidth(value)), value(value),
        nodes(nodes) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Concat; }
  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  ~ConcatNode() override { nodes.clear(); }
  ConcatNode(const ConcatNode &other) = default;
  ConcatNode &operator=(const ConcatNode &other) = default;

  void map(DenseMap<InputNode *, Node *> &nodeToNewNode) override {
    for (auto &node : nodes)
      if (auto *newNode = dyn_cast<InputNode>(node))
        if (auto *newEdgeNode = nodeToNewNode[newNode])
          node = newEdgeNode;
  }
  Node *query(size_t bitOffset) override {
    // TODO: bisect
    for (auto node : nodes) {
      if (bitOffset < node->getWidth())
        return node->query(bitOffset);
      bitOffset -= node->getWidth();
    }
    assert(false && "bit offset is out of range");
    return nullptr;
  }
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent) << "(concat node " << value << "\n";
    for (auto node : nodes) {
      node->dump(indent + 2);
    }
    llvm::dbgs().indent(indent) << ")";
  }
};

struct ReplicateNode : Node {
  Value value;
  Node *node;
  ReplicateNode(Graph *graph, Value value, Node *node)
      : Node(graph, Kind::Replicate, getBitWidth(value)), value(value),
        node(node) {
    assert(node->getWidth() == value.getDefiningOp<comb::ReplicateOp>()
                                   .getInput()
                                   .getType()
                                   .getIntOrFloatBitWidth() &&
           "replicate node width must match the input width");
  }
  static bool classof(const Node *e) { return e->getKind() == Kind::Replicate; }
  ~ReplicateNode() = default;
  ReplicateNode(const ReplicateNode &other) = default;
  ReplicateNode &operator=(const ReplicateNode &other) = default;
  Node *query(size_t bitOffset) override {
    return node->query(bitOffset % node->getWidth());
  }
  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent) << "(replicate node " << value << "\n";
    node->dump(indent + 2);
    llvm::dbgs().indent(indent) << ")";
  }
};

struct ExtractNode : Node {
  Value value;
  size_t lowBit;
  Node *input;
  ExtractNode(const ExtractNode &other) = default;
  ExtractNode &operator=(const ExtractNode &other) = default;
  ExtractNode(Graph *graph, Value value, size_t lowBit, Node *input)
      : Node(graph, Kind::Extract, getBitWidth(value)), value(value),
        lowBit(lowBit), input(input) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Extract; }
  void walk(llvm::function_ref<void(Node *)>) override;
  void walkPreOrder(llvm::function_ref<bool(Node *)>) override;
  ~ExtractNode() = default;

  Node *query(size_t bitOffset) override {
    // exract(a, 2) : i4 -> query(2)
    return input->query(bitOffset + lowBit);
  }
  void dump(size_t indent) const override {
    llvm::dbgs().indent(indent)
        << "(extract node " << value << " " << lowBit << "\n";
    input->dump(indent + 2);
    llvm::dbgs().indent(indent) << ")";
  }
};

struct Graph {
  Graph(hw::HWModuleOp mod) : theModule(mod) {}
  SmallVector<Value>
      inputValues; // Ports + Instance outputs + Registers outputs.
  SmallVector<Operation *>
      outputOperations; // HW output, Instance inputs, Register inputs.

  DenseMap<std::tuple<Operation *, size_t, size_t>, OutputNode *> outputNodes;
  DenseMap<Value, Node *> valueToNodes;
  LogicalResult commitToHWModule(hw::HWModuleOp mod);
  hw::HWModuleOp theModule;
  LogicalResult buildGraph();

  hw::HWModuleOp getModule() { return theModule; }

  SetVector<OutputNode *> locallyClosedOutputs;
  SetVector<OutputNode *> openOutputs;
  SetVector<InputNode *> locallyClosedInputs;

  struct LocalPath {
    OutputNode *fanOut;
    LocalPath(OutputNode *fanOut) : fanOut(fanOut) {}
  };

  SmallVector<LocalPath> localPaths;
  static bool isLocalInput(Value value) {
    return value.getDefiningOp() && !isa<hw::InstanceOp>(value.getDefiningOp());
  }
  static bool isLocalOutput(Operation *op) {
    return !isa<hw::InstanceOp>(op) && !isa<hw::OutputOp>(op);
  }

  void accumulateLocalPaths(OutputNode *outputNode) {
    bool isClosed =
        llvm::all_of(outputNode->getComputedResult(), [&](auto &pair) {
          return isLocalInput(pair.second->value);
        });

    if (isClosed) {
      locallyClosedOutputs.insert(outputNode);
      openOutputs.remove(outputNode);
    } else {
      openOutputs.insert(outputNode);
    }
  }
  void accumulateLocalPaths(Operation *outOp) {
    for (auto [idx, op] : llvm::enumerate(outOp->getOperands())) {
      size_t width = getBitWidth(op);
      auto *inNode = getOrConstant(op);
      for (size_t i = 0; i < width; ++i) {
        auto outputNode = outputNodes.at({outOp, idx, i});
        accumulateLocalPaths(outputNode);
      }
    }
  }
  void accumulateLocalPaths() {
    for (auto outOp : outputOperations) {
      // These are not local paths.
      if (!isLocalOutput(outOp)) {
        continue;
      }

      accumulateLocalPaths(outOp);
    }

    SmallVector<OutputNode *> tmp(openOutputs.begin(), openOutputs.end());
    for (auto outputNode : tmp) {
      accumulateLocalPaths(outputNode);
    }
  }

  LogicalResult
  inlineGraph(ArrayRef<std::pair<hw::InstanceOp, Graph *>> children,
              circt::igraph::InstancePathCache *instancePathCache) {
    // We need to clone the subgraph of the child graph which are reachable from
    // input and output nodes.
    DenseMap<InputNode *, Node *> nodeToNewNode;
    // This stores the cloned child graph.
    DenseMap<Node *, Node *> clonedResult;
    hw::InstanceOp instance;
    size_t clonedNum = 0;
    SmallVector<int> dist(7);
    std::function<Node *(Node *)> recurse = [&](Node *node) -> Node * {
      if (clonedResult.count(node))
        return clonedResult[node];
      clonedNum++;
      if (clonedNum % 1000 == 0) {
        llvm::errs() << clonedNum << "\n";
        for (size_t d : dist) {
          llvm::errs() << d << " ";
        }
        llvm::errs() << "\n";
      }

      Node *result =
          TypeSwitch<Node *, Node *>(node)
              .Case<InputNode>([&](InputNode *node) {
                dist[0]++;
                return allocateNode<InputNode>(node->value, node->bitPos);
              })
              .Case<OutputNode>([&](OutputNode *node) {
                dist[1]++;
                assert(node->node);
                node->node->dump();

                return allocateNode<OutputNode>(node->op, node->operandIdx,
                                                node->bitPos,
                                                recurse(node->node));
              })
              .Case<ConstantNode>([&](ConstantNode *node) {
                dist[2]++;
                return allocateNode<ConstantNode>(node->width, &dummy);
              })
              .Case<DelayNode>([&](DelayNode *node) {
                dist[3]++;
                auto delayNode = allocateNode<DelayNode>(node->getValue(),
                                                         node->getBitPos());
                for (auto [delay, inputNode] : node->getEdges()) {
                  delayNode->addEdge(recurse(inputNode), delay);
                }
                return delayNode;
              })
              .Case<ConcatNode>([&](ConcatNode *node) {
                SmallVector<Node *> nodes;
                dist[4]++;

                for (auto child : node->nodes)
                  nodes.push_back(recurse(child));
                return allocateNode<ConcatNode>(node->value, nodes);
              })
              .Case<ExtractNode>([&](ExtractNode *node) {
                dist[5]++;

                return allocateNode<ExtractNode>(node->value, node->lowBit,
                                                 recurse(node->input));
              })
              .Case<ReplicateNode>([&](ReplicateNode *node) {
                dist[6]++;
                return allocateNode<ReplicateNode>(node->value,
                                                   recurse(node->node));
              });
      auto newPath = instancePathCache->prependInstance(instance, node->path);
      result->setPath(newPath);
      clonedResult[node] = result;
      return result;
    };

    for (auto [instanceOp, childGraph] : children) {
      // Inline the child graph into the parent graph.
      llvm::dbgs() << "Inlining " << childGraph->theModule.getModuleName()
                   << " to " << theModule.getModuleName() << "\n";
      auto childArgs = childGraph->theModule.getBodyBlock()->getArguments();
      auto childOutputs = childGraph->theModule.getBodyBlock()->getTerminator();

      clonedResult.clear();
      instance = instanceOp;
      for (auto [operand, childArg] :
           llvm::zip(instanceOp->getOpOperands(), childArgs)) {
        size_t width = getBitWidth(childArg);
        auto *inputNodes = childGraph->getOrConstant(childArg);
        for (size_t i = 0; i < width; ++i) {
          auto *outputNode = outputNodes.at(
              {operand.getOwner(), operand.getOperandNumber(), i});
          clonedResult[cast<InputNode>(inputNodes->query(i))] =
              outputNode->query(0); // (この中にinstanceがいる)
        }
      }

      for (auto [instanceResult, childOutput] :
           llvm::zip(instanceOp.getResults(), childOutputs->getOpOperands())) {

        size_t width = getBitWidth(childOutput.get());
        llvm::dbgs() << " Inlining instance result " << width << "\n";
        for (size_t i = 0; i < width; ++i) {
          auto *outputNode = childGraph->outputNodes.at(
              {childOutput.getOwner(), childOutput.getOperandNumber(), i});
          recurse(outputNode);
          auto inputNode =
              cast<InputNode>(valueToNodes.at(instanceResult)->query(i));
          LLVM_DEBUG({
            llvm::dbgs() << "Updated input node: " << instanceResult << "\n";
            inputNode->dump(0);
            llvm::dbgs() << " @ " << inputNode << "\n";
          });

          nodeToNewNode[inputNode] = clonedResult[outputNode]->query(0);
        }
      }
      llvm::dbgs() << "Done Inlining " << childGraph->theModule.getModuleName()
                   << " to " << theModule.getModuleName() << "\n";

      for (auto outputNode : childGraph->openOutputs) {
        recurse(outputNode);
        openOutputs.insert(cast<OutputNode>(clonedResult.at(outputNode)));
      }
    }
    llvm::dbgs() << "handle output operatoins";

    DenseSet<Node *> visited;

    for (auto op : outputOperations) {
      for (auto &operand : op->getOpOperands()) {
        for (size_t i = 0; i < getBitWidth(operand.get()); ++i) {
          auto outPutNode = outputNodes.at({op, operand.getOperandNumber(), i});
          outPutNode->computedResult.reset();
          outPutNode->walkPreOrder([&](Node *node) {
            LLVM_DEBUG({
              llvm::dbgs() << "Updating node: ";
              node->dump();
              llvm::dbgs() << "\n";
            });
            if (!visited.insert(node).second)
              return false;
            node->map(nodeToNewNode);

            LLVM_DEBUG({
              llvm::dbgs() << "Updated node: ";
              node->dump();
              llvm::dbgs() << "\n";
            });
            return true;
          });
          LLVM_DEBUG({ outPutNode->dump(0); });
          bool found = false;
          /*
          outPutNode->walkPreOrder([&](Node *node) {
            if (auto *inputNode = dyn_cast<InputNode>(node)) {
              if (auto inst =
                      inputNode->value.getDefiningOp<hw::InstanceOp>()) {
                llvm::dbgs() << "Instance: " << inst << "\n";
                inputNode->dump(0);
                llvm::dbgs() << "\n";
                found = true;
              }
            }
          });
          */
          if (found) {
            outPutNode->dump(0);
            return failure();
          }
        }
      }
    }

    for (auto outPutNode : openOutputs)
      outPutNode->walkPreOrder([&](Node *node) {
        if (!visited.insert(node).second)
          return false;
        node->map(nodeToNewNode);
        return true;
      });

    outputOperations.erase(
        llvm::remove_if(outputOperations,
                        [&](Operation *op) { return isa<hw::InstanceOp>(op); }),
        outputOperations.end());
    llvm::errs() << "Done " << theModule.getModuleNameAttr() << '\n';

    return success();
  }

  // llvm::SpecificBumpPtrAllocator<DelayNode> delayAllocator;
  // llvm::SpecificBumpPtrAllocator<ConcatNode> concatAllocator;
  // llvm::SpecificBumpPtrAllocator<ReplicateNode> replicateAllocator;
  // llvm::SpecificBumpPtrAllocator<ExtractNode> extractAllocator;
  // llvm::SpecificBumpPtrAllocator<InputNode> inputAllocator;
  // llvm::SpecificBumpPtrAllocator<OutputNode> outputAllocator;
  // llvm::SpecificBumpPtrAllocator<ConstantNode> constantAllocator;

  // SmallVector<std::unique_ptr<Node>> nodes;

  // For multiple objects:
  // template <typename NodeTy, typename AllocatorTy, typename... Args>
  // NodeTy *allocateAndConstruct(AllocatorTy &allocator, size_t count,
  //                              Args &&...args) {
  //   // Allocate array of objects
  //   NodeTy *nodes = allocator.Allocate(count);

  //   // Construct each object
  //   for (size_t i = 0; i < count; ++i) {
  //     new (&nodes[i]) NodeTy(std::forward<Args>(args)...);
  //   }

  //   return nodes;
  // }
  /*
  template <typename NodeTy, typename AllocatorTy, typename... Args>
  NodeTy *allocateAndConstruct(AllocatorTy &allocator, size_t width = 1,
                               Args &&...args) {
    auto *node = allocator.Allocate(width);
    for (auto i = 0; i < width; ++i) {
      NodeTy nodeTy(std::forward<Args>(args)...);
      node[i] = nodeTy;
    }

    return node;
  }*/
  void addInputNode(Value value) {
    inputValues.push_back(value);
    auto width = getBitWidth(value);
    assert(isRootValue(value) && "only root values can be input nodes");
    SmallVector<Node *> nodes;
    for (auto i = 0; i < width; ++i) {
      auto nodePtr = allocateNode<InputNode>(value, i);
      nodes.push_back(nodePtr);
    }

    // Why not ArrayRef(node, width) works?
    setResultValue(value, nodes);
  }

  void setResultValue2(Value value, Node *node) {
    auto result = valueToNodes.insert(std::make_pair(value, node));
    assert(result.second && "value already exists");
  }

  ConstantNode dummy{this, 0, nullptr};
  void addConstantNode(Value value) {
    auto width = hw::getBitWidth(value.getType());
    auto *nodePtr = allocateNode<ConstantNode>(width < 0 ? 1 : width, &dummy);
    setResultValue2(value, nodePtr);
  }

  Node *getOrConstant(Value value) {
    auto it = valueToNodes.find(value);
    if (it != valueToNodes.end())
      return it->second;

    value.getDefiningOp()->emitWarning() << "is ignored\n";

    return valueToNodes[value] =
               allocateNode<ConstantNode>(getBitWidth(value), &dummy);
  }

  void addOutputNode(OpOperand &operand) {
    auto operandOp = operand.getOwner();
    size_t width = getBitWidth(operand.get());
    for (size_t i = 0; i < width; ++i) {
      auto *outputNode =
          allocateNode<OutputNode>(operandOp, operand.getOperandNumber(), i,
                                   getOrConstant(operand.get())->query(i));
      outputNodes.insert(
          {{operandOp, operand.getOperandNumber(), i}, outputNode});
      outputNode->getComputedResult();
    }
  }

  // FIXME: SpecificBumpPtrAllocator crashes for some reason. For now use
  // unique ptrs.

  template <typename NodeTy, typename... Args>
  NodeTy *allocateNode(Args &&...args) {
    auto nodePtr = std::make_unique<NodeTy>(this, std::forward<Args>(args)...);
    nodePool.push_back(std::move(nodePtr));
    return cast<NodeTy>(nodePool.back().get());
  }

  SmallVector<std::unique_ptr<Node>> nodePool;

  void setResultValue(Value value, Node *nodePtr) {
    auto result = valueToNodes.insert(std::make_pair(value, nodePtr));
    assert(result.second && "value already exists");
  }

  void setResultValue(Value value, ArrayRef<Node *> nodes) {
    if (getBitWidth(value) == 1 || nodes.size() == 1) {
      auto result = valueToNodes.insert(std::make_pair(value, nodes[0]));
      assert(result.second && "value already exists");
      return;
    }

    auto nodePtr = allocateNode<ConcatNode>(value, nodes);
    auto result = valueToNodes.insert(std::make_pair(value, nodePtr));
    assert(result.second && "value already exists");
  }

  LogicalResult addDelayNode(Value value, mlir::OperandRange inputs) {
    size_t width = value.getType().getIntOrFloatBitWidth();
    // auto *nodes =
    //     allocateAndConstruct<DelayNode>(delayAllocator, width, value, 0);
    SmallVector<Node *> nodePtrs;
    for (auto i = 0; i < width; ++i) {
      auto nodePtr = allocateNode<DelayNode>(value, i);
      for (auto operand : inputs) {
        if (valueToNodes.find(operand) == valueToNodes.end()) {
          value.getDefiningOp()->emitWarning() << operand << "is skipped";
          continue;
        }
        auto *inputNode = valueToNodes[operand];
        assert(inputNode && "input node not found");

        auto *node = inputNode->query(i);
        assert(node);
        uint64_t delay = std::max(1u, llvm::Log2_64_Ceil(inputs.size()));
        nodePtr->addEdge(node, delay);
      }

      (void)nodePtr->computeResult();

      nodePtrs.push_back(nodePtr);
    }

    setResultValue(value, nodePtrs);

    return success();
  }

  template <typename ty>
  LogicalResult addConcatNode(Value value, ty inputs) {
    size_t width = getBitWidth(value);
    SmallVector<Node *> nodes;
    for (auto operand : llvm::reverse(inputs)) {
      auto *inputNode = getOrConstant(operand);
      // llvm::dbgs() << "inputNode: " << operand << "\n";
      assert(inputNode && "input node not found");
      nodes.push_back(inputNode);
    }

    setResultValue(value, nodes);
    return success();
  }

  LogicalResult addExtractNode(Value value, Value operand, size_t lowBit) {
    auto *inputNode = getOrConstant(operand);
    if (!inputNode) {
      llvm::errs() << operand << '\n';
    }
    assert(inputNode && "input node not found");
    // inputNode->dump();
    auto *node = allocateNode<ExtractNode>(value, lowBit, inputNode);
    assert(node);

    setResultValue2(value, node);
    return success();
  }

  LogicalResult addReplicateNode(Value value, Value operand) {
    auto *inputNode = getOrConstant(operand);
    assert(inputNode && "input node not found");
    auto *node = allocateNode<ReplicateNode>(value, inputNode);
    assert(node && "node allocation failed");
    setResultValue2(value, node);
    return success();
  }
};

LogicalResult Graph::commitToHWModule(hw::HWModuleOp mod) { return success(); }

LogicalResult Graph::buildGraph() {
  auto &graph = *this;
  for (auto arg : theModule.getBodyBlock()->getArguments()) {
    addInputNode(arg);
  }

  SmallVector<Operation *> interestingOps;
  // Add input nodes.
  theModule.walk([&](Operation *op) {
    if (isa<seq::FirRegOp, seq::CompRegOp, hw::InstanceOp, seq::FirMemReadOp>(
            op)) {
      for (auto result : op->getResults())
        graph.addInputNode(result);
    } else if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
                   aig::AndInverterOp, hw::WireOp>(op)) {
      interestingOps.push_back(op);
    } else if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      interestingOps.push_back(op);
    } else if (isa<sv::ConstantXOp>(op)) {
      interestingOps.push_back(op);
    }
    if (isa<seq::FirRegOp, seq::CompRegOp, hw::InstanceOp, hw::OutputOp>(op)) {
      graph.outputOperations.push_back(op);
    }
  });

  auto isOperandReady = [&](Value value, Operation *op) {
    return isRootValue(value);
  };
  llvm::errs() << theModule.getModuleNameAttr() << "Running Toposort\n";
  mlir::computeTopologicalSorting(interestingOps, isOperandReady);

  llvm::errs() << theModule.getModuleNameAttr() << "Running Graph const";
  SmallVector<int64_t> cnt(8);

  for (auto op : interestingOps) {
    if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
      graph.addDelayNode(andInverterOp, andInverterOp.getInputs());
    } else if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      graph.addConcatNode(concatOp, concatOp.getInputs());
    } else if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
      graph.addExtractNode(extractOp, extractOp.getInput(),
                           extractOp.getLowBit());

    } else if (auto replicateOp = dyn_cast<comb::ReplicateOp>(op)) {
      graph.addReplicateNode(replicateOp, replicateOp.getInput());

    } else if (op->hasTrait<mlir::OpTrait::ConstantLike>() ||
               (isa<sv::ConstantXOp>(op))) {
      graph.addConstantNode(op->getResult(0));

    } else if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
      graph.addConcatNode(wireOp, ArrayRef<Value>{wireOp.getInput()});
    }
  }

  for (auto outputOp : graph.outputOperations) {
    for (auto &operand : outputOp->getOpOperands()) {
      graph.addOutputNode(operand);
    }
  }

  // %0 = DelayNode(a, b, c)
  // %1 = aig.cut.node()
  // %2 = aig.cut.node()

  bool modifyModule = true;
  if (modifyModule) {
    OpBuilder builder(theModule.getContext());
    DenseMap<InputNode *, Value> inputNodeToValue;
    for (auto op : graph.outputOperations) {
      builder.setInsertionPoint(op);
      size_t idx = 0;
      for (auto operand : llvm::make_early_inc_range(op->getOperands())) {
        auto guard = llvm::make_scope_exit([&]() { ++idx; });
        auto *node = graph.getOrConstant(operand);
        SmallVector<Value> operands;
        for (auto i = 0; i < node->getWidth(); ++i) {
          auto *result = node->query(i);
          result->walk([&](Node *node) {
            if (auto *delayNode = dyn_cast<DelayNode>(node))
              delayNode->shrinkEdges();
          });
          assert(result && "result not found");
          SmallVector<std::pair<int64_t, InputNode *>> delays;
          result->populateResults(delays);
          SmallVector<Value> delayOperands;
          SmallVector<int64_t> operandDelays;
          for (auto [delay, inputNode] : delays) {
            auto &value = inputNodeToValue[inputNode];
            if (!value) {
              value = getBitWidth(inputNode->value) == 1
                          ? inputNode->value
                          : builder.createOrFold<comb::ExtractOp>(
                                op->getLoc(), inputNode->value,
                                inputNode->getBitPos(), 1);
            }
            operandDelays.push_back(delay);
            delayOperands.push_back(value);
          }

          operands.push_back(builder.create<aig::DelayOp>(
              op->getLoc(),
              isa<seq::ClockType>(operand.getType()) ? operand.getType()
                                                     : builder.getI1Type(),
              delayOperands, operandDelays));
        }
        if (operands.size() == 1) {
          op->setOperand(idx, operands.front());
          continue;
        }

        std::reverse(operands.begin(), operands.end());
        auto delayOp = builder.createOrFold<comb::ConcatOp>(
            op->getLoc(), operand.getType(), operands);
        op->setOperand(idx, delayOp);
      }
    }
  }

  return success();
}

// Extract must be up. Concat must be down.

// ExtractOp(Concat(a, b, c)) -> Concat(Extract(b), Extract(c))
// ExtractOp(AndDelay(a, b, c)) -> AndDelay(Extract(b))
// ConcatOp(AndDelay(a, b, c), AndDelay(d, e, f)) -> keep as is
// AndDelay(Concat(delay_a, delay_b, delay_c), Concat(delay_d, delay_e,
// delay_f))
//   -> Concat(AndDelay(delay_a, delay_d), AndDelay(delay_b, delay_e),
//             AndDelay(delay_c, delay_f))

struct LongestPathAnalysisImpl {
  LongestPathAnalysisImpl(mlir::ModuleOp mod,
                          igraph::InstanceGraph *instanceGraph,
                          StringAttr topModuleName)
      : mod(mod), instanceGraph(instanceGraph), topModuleName(topModuleName) {}
  LogicalResult flatten(hw::HWModuleOp mod);
  LogicalResult run();

  mlir::MLIRContext *getContext() { return mod.getContext(); }

private:
  mlir::ModuleOp mod;
  StringAttr topModuleName;
  igraph::InstanceGraph *instanceGraph;
  DenseMap<StringAttr, std::unique_ptr<Graph>> moduleToGraph;
};

LogicalResult LongestPathAnalysisImpl::run() {
  std::mutex mutex;
  SmallVector<hw::HWModuleOp> underHierarchy;
  for (auto *node : llvm::post_order(instanceGraph->lookup(topModuleName)))
    if (node && node->getModule())
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule())) {
        if (hwMod.getModuleName().ends_with("_assert") ||
            hwMod.getModuleName().ends_with("_cover") ||
            hwMod.getModuleName().ends_with("_assume") ||
            hwMod.getNumOutputPorts() == 0)
          continue;

        underHierarchy.push_back(hwMod);
        moduleToGraph[hwMod.getModuleNameAttr()] =
            std::make_unique<Graph>(hwMod);
      }
  llvm::errs() << "Under hierarchy: " << underHierarchy.size() << "\n";

  auto result = mlir::failableParallelForEach(
      getContext(), underHierarchy, [&](hw::HWModuleOp mod) {
        auto startTime = std::chrono::high_resolution_clock::now();
        {
          std::lock_guard<std::mutex> lock(mutex);
          llvm::errs() << mod.getName() << " start\n";
        }
        auto &graph = moduleToGraph.at(mod.getModuleNameAttr());

        auto result = graph->buildGraph();

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime);
        {
          std::lock_guard<std::mutex> lock(mutex);
          llvm::errs() << mod.getName() << " end, time: " << duration.count()
                       << "ms\n";
        }

        return result;
      });

  SmallVector<OutputNode *> results;
  circt::igraph::InstancePathCache instancePathCache(*instanceGraph);
  DenseSet<StringAttr> visited;
  for (auto moduleOp : underHierarchy) {
    auto *node = instanceGraph->lookup(moduleOp.getModuleNameAttr());
    auto &graph = moduleToGraph.at(moduleOp.getModuleNameAttr());
    SmallVector<std::pair<hw::InstanceOp, Graph *>, 4> childInstances;
    for (auto *child : *node) {

      if (!child || !child->getTarget())
        continue;

      auto target = child->getTarget()->getModule();

      if (!target || !child->getInstance())
        continue;

      auto instanceOp = dyn_cast<hw::InstanceOp>(*child->getInstance());
      if (!instanceOp)
        return child->getInstance().emitError() << "unsupported instance";
      // skip external modules.
      auto childGraph = moduleToGraph.find(target.getModuleNameAttr());
      if (childGraph == moduleToGraph.end())
        continue;
      childInstances.push_back({instanceOp, childGraph->second.get()});
    }

    graph->inlineGraph(childInstances, &instancePathCache);

    // Accumulate local paths.
    graph->accumulateLocalPaths();

    llvm::dbgs() << "\nLocally closed node: ";
    llvm::dbgs() << graph->theModule.getModuleName() << "\n";
    llvm::dbgs() << graph->locallyClosedOutputs.size() << "\n";
    llvm::dbgs() << graph->openOutputs.size() << "\n";
    for (auto outputNode : graph->locallyClosedOutputs) {

      // outputNode->print(llvm::dbgs());
      // llvm::dbgs() << "\n";
      results.emplace_back(outputNode);
    }

    visited.insert(moduleOp.getModuleNameAttr());
  }

  auto top = underHierarchy.back();
  auto &topGraph = moduleToGraph.at(top.getModuleNameAttr());

  for (auto &outputOp : topGraph->outputOperations) {
    for (auto &operand : outputOp->getOpOperands()) {
      for (size_t i = 0; i < getBitWidth(operand.get()); ++i) {
        auto *output =
            topGraph->outputNodes.at({outputOp, operand.getOperandNumber(), i});
        results.emplace_back(output);
      }
    }
  }

  for (auto &outputNode : topGraph->openOutputs) {
    results.emplace_back(outputNode);
  }

  SmallVector<llvm::json::Object> resultsJSON;
  size_t topK = 100;

  SmallVector<std::tuple<int64_t, OutputNode *, InputNode *>> longestPaths;
  // std::priority_queue<std::tuple<int64_t, OutputNode *, InputNode *>> queue;
  for (auto &outputNode : results) {
    // resultsJSON.push_back(outputNode->getJSONObject());
    for (auto [length, inputNode] : outputNode->getComputedResult()) {
      longestPaths.emplace_back(length, outputNode, inputNode);
    }
  }

  std::sort(longestPaths.begin(), longestPaths.end(),
            std::greater<std::tuple<int64_t, OutputNode *, InputNode *>>());

  llvm::errs() << "// ------ LongestPathAnalysis Summary -----\n";
  llvm::errs() << "Top module: " << topModuleName << "\n";
  llvm::errs() << "Top " << topK << " longest paths:\n";
  for (size_t i = 0; i < std::min(topK, longestPaths.size()); ++i) {
    auto [length, outputNode, inputNode] = longestPaths[i];
    llvm::errs() << i + 1 << ": length = " << length << " ";
    outputNode->print(llvm::errs());
    llvm::errs() << " -> ";
    inputNode->print(llvm::errs());
    llvm::errs() << "\n";
  }

  return result;
}

namespace {
struct PrintLongestPathAnalysisPass
    : public impl::PrintLongestPathAnalysisBase<PrintLongestPathAnalysisPass> {
  using PrintLongestPathAnalysisBase::PrintLongestPathAnalysisBase;
  LogicalResult rewrite(mlir::ModuleOp mod,
                        igraph::InstanceGraph &instanceGraph);
  LogicalResult rewrite(hw::HWModuleOp mod);

  using PrintLongestPathAnalysisBase::outputJSONFile;
  using PrintLongestPathAnalysisBase::printSummary;
  using PrintLongestPathAnalysisBase::topModuleName;
  void runOnOperation() override;
};

} // namespace

void PrintLongestPathAnalysisPass::runOnOperation() {
  if (topModuleName.empty()) {
    getOperation().emitError()
        << "'top-name' option is required for PrintLongestPathAnalysis";
    return signalPassFailure();
  }

  auto &symTbl = getAnalysis<mlir::SymbolTable>();
  auto &instanceGraph = getAnalysis<igraph::InstanceGraph>();
  auto top = symTbl.lookup<hw::HWModuleOp>(topModuleName);
  if (!top) {
    getOperation().emitError()
        << "top module '" << topModuleName << "' not found";
    return signalPassFailure();
  }

  bool isInplace = true;
  Block *workSpaceBlock;
  OpBuilder builder(&getContext());
  mlir::ModuleOp workSpaceMod;
  if (isInplace) {
    workSpaceBlock = getOperation().getBody();
    builder.setInsertionPointToStart(workSpaceBlock);
    workSpaceMod = getOperation();
  } else {
    llvm::errs() << "Clone subhierarchy\n";
    workSpaceBlock = new Block();
    builder.setInsertionPointToStart(workSpaceBlock);
    workSpaceMod = builder.create<mlir::ModuleOp>(
        UnknownLoc::get(&getContext()), "workSpace");
    builder.setInsertionPointToStart(workSpaceMod.getBody());
    // Clone subhierarchy under top module.
    for (auto *node : llvm::post_order(instanceGraph.lookup(top))) {
      if (node && node->getModule())
        if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule())) {
          builder.clone(*hwMod);
        }
    }
    llvm::errs() << "Cloned subhierarchy\n";
  }

  LongestPathAnalysisImpl longestPathAnalysis(
      workSpaceMod, &instanceGraph,
      StringAttr::get(&getContext(), topModuleName));
  if (failed(longestPathAnalysis.run())) {
    llvm::errs() << "Failed to run longest path analysis\n";
    return signalPassFailure();
  }

  // TODO: Clean up symbol uses in the cloned subhierarchy.

  if (printSummary) {
    llvm::errs() << "// ------ LongestPathAnalysis Summary -----\n";
    llvm::errs() << "Top module: " << topModuleName << "\n";
  }

  if (!outputJSONFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputJSONFile, ec);
    if (ec) {
      emitError(UnknownLoc::get(&getContext()))
          << "failed to open output JSON file '" << outputJSONFile
          << "': " << ec.message();
      return signalPassFailure();
    }
  }

  if (!isInplace) {
    workSpaceBlock->erase();
  }
}

void InputNode::walk(llvm::function_ref<void(Node *)> callback) {
  callback(this);
}

void InputNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  callback(this);
}

void OutputNode::walk(llvm::function_ref<void(Node *)> callback) {
  node->walk(callback);
  callback(this);
}

void OutputNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  if (callback(this))
    node->walkPreOrder(callback);
}

void DelayNode::walk(llvm::function_ref<void(Node *)> callback) {
  for (auto [delay, inputNode] : edges) {
    inputNode->walk(callback);
  }

  callback(this);
}

void DelayNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  if (!callback(this))
    return;
  for (auto [delay, inputNode] : edges) {
    inputNode->walkPreOrder(callback);
  }
}

void ConcatNode::walk(llvm::function_ref<void(Node *)> callback) {
  for (auto node : nodes) {
    node->walk(callback);
  }
  callback(this);
}

void ConcatNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  if (callback(this))
    for (auto node : nodes) {
      node->walkPreOrder(callback);
    }
}

void ExtractNode::walk(llvm::function_ref<void(Node *)> callback) {
  input->walk(callback);
  callback(this);
}

void ExtractNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  if (callback(this))
    input->walkPreOrder(callback);
}

void ReplicateNode::walk(llvm::function_ref<void(Node *)> callback) {
  node->walk(callback);
  callback(this);
}

void ReplicateNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  if (callback(this))
    node->walkPreOrder(callback);
}

void ConstantNode::walk(llvm::function_ref<void(Node *)> callback) {
  callback(this);
  if (boolNode)
    boolNode->walk(callback);
}

void ConstantNode::walkPreOrder(llvm::function_ref<bool(Node *)> callback) {
  callback(this);
  if (boolNode)
    boolNode->walkPreOrder(callback);
}
