//===- TimingGraph.cpp - Timing Graph Implementation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/TimingGraph.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "timing-graph"

using namespace circt;
using namespace circt::synth::timing;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static size_t getBitWidth(Value value) {
  auto type = value.getType();
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  return 1;
}

static bool isSequentialOp(Operation *op) {
  return isa<seq::CompRegOp, seq::FirRegOp, seq::FirMemReadOp,
             seq::FirMemReadWriteOp>(op);
}

//===----------------------------------------------------------------------===//
// TimingGraph Implementation
//===----------------------------------------------------------------------===//

TimingGraph::TimingGraph(hw::HWModuleOp module) : module(module) {}

TimingGraph::~TimingGraph() = default;

TimingNodeId TimingGraph::createNode(Value value, uint32_t bitPos,
                                     TimingNodeKind kind, StringRef name) {
  TimingNodeId id{static_cast<uint32_t>(nodes.size())};
  auto node = std::make_unique<TimingNode>(id, value, bitPos, kind, name);

  // Track start/end points
  if (node->isStartPoint())
    startPoints.push_back(node.get());
  if (node->isEndPoint())
    endPoints.push_back(node.get());

  // Add to lookup map
  valueToNode[{value, bitPos}] = node.get();

  nodes.push_back(std::move(node));
  return id;
}

TimingArc *TimingGraph::createArc(TimingNode *from, TimingNode *to,
                                  int64_t delay) {
  auto arc = std::make_unique<TimingArc>(from, to, delay);
  TimingArc *arcPtr = arc.get();

  from->addFanout(arcPtr);
  to->addFanin(arcPtr);

  arcs.push_back(std::move(arc));
  return arcPtr;
}

TimingNode *TimingGraph::findNode(Value value, uint32_t bitPos) const {
  auto it = valueToNode.find({value, bitPos});
  if (it != valueToNode.end())
    return it->second;
  return nullptr;
}

TimingNode *TimingGraph::getOrCreateNode(Value value, uint32_t bitPos) {
  if (auto *existing = findNode(value, bitPos))
    return existing;

  // Determine the node kind
  TimingNodeKind kind = TimingNodeKind::Combinational;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == module.getBodyBlock())
      kind = TimingNodeKind::InputPort;
  } else if (auto *defOp = value.getDefiningOp()) {
    if (isSequentialOp(defOp))
      kind = TimingNodeKind::RegisterOutput;
  }

  std::string name = getNameForValue(value);
  createNode(value, bitPos, kind, name);
  return findNode(value, bitPos);
}

std::string TimingGraph::getNameForValue(Value value) {
  // Try to get a name from the defining op
  if (auto *defOp = value.getDefiningOp()) {
    // Check for sv.namehint attribute
    if (auto nameAttr = defOp->getAttrOfType<StringAttr>("sv.namehint"))
      return nameAttr.getValue().str();

    // Check for hw.name attribute
    if (auto nameAttr = defOp->getAttrOfType<StringAttr>("hw.name"))
      return nameAttr.getValue().str();

    // For registers, use the name attribute
    if (auto compreg = dyn_cast<seq::CompRegOp>(defOp)) {
      if (auto nameAttr = compreg.getNameAttr())
        return nameAttr.getValue().str();
    }
    if (auto firreg = dyn_cast<seq::FirRegOp>(defOp)) {
      if (auto nameAttr = firreg.getNameAttr())
        return nameAttr.getValue().str();
    }

    // For wires
    if (auto wire = dyn_cast<hw::WireOp>(defOp)) {
      if (auto nameAttr = wire.getNameAttr())
        return nameAttr.getValue().str();
    }

    // Default: use op name
    return defOp->getName().getStringRef().str();
  }

  // Block argument (port)
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner() == module.getBodyBlock()) {
      size_t argNum = blockArg.getArgNumber();
      return module.getInputName(argNum).str();
    }
  }

  return "unnamed";
}

int64_t TimingGraph::getDelayCost(Operation *op) const {
  // AIG operations
  if (isa<aig::AndInverterOp>(op))
    return 1;

  // Comb operations
  if (isa<comb::MuxOp>(op))
    return 1;
  if (auto andOp = dyn_cast<comb::AndOp>(op))
    return llvm::Log2_64_Ceil(andOp.getNumOperands());
  if (auto orOp = dyn_cast<comb::OrOp>(op))
    return llvm::Log2_64_Ceil(orOp.getNumOperands());
  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return llvm::Log2_64_Ceil(xorOp.getNumOperands());

  // Zero-cost operations (bit manipulation)
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op))
    return 0;

  // Default cost
  return 1;
}

LogicalResult TimingGraph::build() {
  LLVM_DEBUG(llvm::dbgs() << "Building timing graph for module: "
                          << module.getModuleName() << "\n");

  // First pass: create nodes for all values
  // Process input ports
  for (auto arg : module.getBodyBlock()->getArguments()) {
    size_t width = getBitWidth(arg);
    for (size_t bit = 0; bit < width; ++bit)
      getOrCreateNode(arg, bit);
  }

  // Process operations
  for (auto &op : module.getBodyBlock()->getOperations()) {
    if (failed(processOperation(&op)))
      return failure();
  }

  // Compute topological order for traversal
  computeTopologicalOrder();

  LLVM_DEBUG(llvm::dbgs() << "Timing graph built: " << nodes.size()
                          << " nodes, " << arcs.size() << " arcs, "
                          << startPoints.size() << " start points, "
                          << endPoints.size() << " end points\n");

  return success();
}

LogicalResult TimingGraph::processOperation(Operation *op) {
  // Skip constants
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();

  // Handle output operation - create end points for output ports
  if (auto outputOp = dyn_cast<hw::OutputOp>(op)) {
    for (auto [idx, operand] : llvm::enumerate(outputOp.getOperands())) {
      size_t width = getBitWidth(operand);
      for (size_t bit = 0; bit < width; ++bit) {
        // Create output port node
        std::string name = module.getOutputName(idx).str();
        TimingNodeId id{static_cast<uint32_t>(nodes.size())};
        auto node = std::make_unique<TimingNode>(
            id, operand, bit, TimingNodeKind::OutputPort, name);
        endPoints.push_back(node.get());

        // Create arc from operand to output
        if (auto *fromNode = findNode(operand, bit))
          createArc(fromNode, node.get(), 0);

        nodes.push_back(std::move(node));
      }
    }
    return success();
  }

  // Handle sequential elements - they are both end and start points
  if (isSequentialOp(op)) {
    // The register output is a start point (already handled in getOrCreateNode)
    for (auto result : op->getResults()) {
      size_t width = getBitWidth(result);
      for (size_t bit = 0; bit < width; ++bit)
        getOrCreateNode(result, bit);
    }

    // The register input is an end point
    if (auto compreg = dyn_cast<seq::CompRegOp>(op)) {
      Value input = compreg.getInput();
      size_t width = getBitWidth(input);
      for (size_t bit = 0; bit < width; ++bit) {
        std::string name;
        if (auto nameAttr = compreg.getNameAttr())
          name = nameAttr.getValue().str() + "_D";
        else
          name = "reg_D";
        TimingNodeId id{static_cast<uint32_t>(nodes.size())};
        auto node = std::make_unique<TimingNode>(
            id, input, bit, TimingNodeKind::RegisterInput, name);
        endPoints.push_back(node.get());

        if (auto *fromNode = findNode(input, bit))
          createArc(fromNode, node.get(), 0);

        nodes.push_back(std::move(node));
      }
    } else if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
      Value input = firreg.getNext();
      size_t width = getBitWidth(input);
      for (size_t bit = 0; bit < width; ++bit) {
        std::string name;
        if (auto nameAttr = firreg.getNameAttr())
          name = nameAttr.getValue().str() + "_D";
        else
          name = "reg_D";
        TimingNodeId id{static_cast<uint32_t>(nodes.size())};
        auto node = std::make_unique<TimingNode>(
            id, input, bit, TimingNodeKind::RegisterInput, name);
        endPoints.push_back(node.get());

        if (auto *fromNode = findNode(input, bit))
          createArc(fromNode, node.get(), 0);

        nodes.push_back(std::move(node));
      }
    }
    return success();
  }

  // Handle combinational operations
  int64_t delay = getDelayCost(op);

  for (auto result : op->getResults()) {
    size_t resultWidth = getBitWidth(result);
    for (size_t bit = 0; bit < resultWidth; ++bit) {
      auto *toNode = getOrCreateNode(result, bit);

      // Create arcs from operands
      for (auto operand : op->getOperands()) {
        size_t operandWidth = getBitWidth(operand);
        // For simplicity, connect all operand bits to result bits
        // More sophisticated handling could be added for extract/concat
        size_t srcBit = std::min(bit, operandWidth - 1);
        if (auto *fromNode = findNode(operand, srcBit))
          createArc(fromNode, toNode, delay);
      }
    }
  }

  return success();
}

void TimingGraph::computeTopologicalOrder() {
  // Simple BFS-based topological sort from start points
  topoOrder.clear();
  llvm::DenseSet<TimingNode *> visited;
  llvm::SmallVector<TimingNode *, 64> worklist;

  // Start from all start points
  for (auto *node : startPoints) {
    worklist.push_back(node);
    visited.insert(node);
  }

  while (!worklist.empty()) {
    auto *node = worklist.pop_back_val();
    topoOrder.push_back(node);

    for (auto *arc : node->getFanout()) {
      auto *successor = arc->getTo();
      if (visited.insert(successor).second)
        worklist.push_back(successor);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Topological order: " << topoOrder.size()
                          << " nodes\n");
}

