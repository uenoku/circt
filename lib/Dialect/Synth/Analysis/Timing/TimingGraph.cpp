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
#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
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

static std::string makeHierarchicalName(StringRef contextPath,
                                        StringRef localName) {
  if (contextPath.empty())
    return localName.str();
  return (contextPath + "/" + localName).str();
}

static std::string makeChildContext(StringRef parent, StringRef childInst) {
  if (parent.empty())
    return childInst.str();
  return (parent + "/" + childInst).str();
}

static StringAttr getContextAttr(MLIRContext *ctx, StringRef contextPath) {
  if (contextPath.empty())
    return {};
  return StringAttr::get(ctx, contextPath);
}

//===----------------------------------------------------------------------===//
// TimingGraph Implementation
//===----------------------------------------------------------------------===//

TimingGraph::TimingGraph(hw::HWModuleOp module) : module(module) {}

TimingGraph::TimingGraph(mlir::ModuleOp circuit, hw::HWModuleOp topModule)
    : circuit(circuit), module(topModule), hierarchical(true) {}

TimingGraph::~TimingGraph() = default;

TimingNodeId TimingGraph::createNode(Value value, uint32_t bitPos,
                                     TimingNodeKind kind, StringRef name,
                                     StringRef contextPath, bool addToLookup) {
  TimingNodeId id{static_cast<uint32_t>(nodes.size())};
  auto node = std::make_unique<TimingNode>(
      id, value, bitPos, kind, makeHierarchicalName(contextPath, name));

  // Track start/end points
  if (node->isStartPoint())
    startPoints.push_back(node.get());
  if (node->isEndPoint())
    endPoints.push_back(node.get());

  // Add to lookup map
  if (addToLookup)
    valueToNode[{getContextAttr(module->getContext(), contextPath), value,
                 bitPos}] = node.get();

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
  return findNode(value, bitPos, "");
}

TimingNode *TimingGraph::findNode(Value value, uint32_t bitPos,
                                  StringRef contextPath) const {
  auto it = valueToNode.find(
      {getContextAttr(module->getContext(), contextPath), value, bitPos});
  if (it != valueToNode.end())
    return it->second;
  return nullptr;
}

TimingNode *TimingGraph::getOrCreateNode(Value value, uint32_t bitPos,
                                         hw::HWModuleOp currentModule,
                                         StringRef contextPath,
                                         bool topContext) {
  if (auto *existing = findNode(value, bitPos, contextPath))
    return existing;

  // Determine the node kind
  TimingNodeKind kind = TimingNodeKind::Combinational;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (topContext && blockArg.getOwner() == currentModule.getBodyBlock())
      kind = TimingNodeKind::InputPort;
  } else if (auto *defOp = value.getDefiningOp()) {
    if (isSequentialOp(defOp))
      kind = TimingNodeKind::RegisterOutput;
  }

  std::string name = getNameForValue(value, currentModule, contextPath);
  createNode(value, bitPos, kind, name, contextPath);
  return findNode(value, bitPos, contextPath);
}

std::string TimingGraph::getNameForValue(Value value,
                                         hw::HWModuleOp currentModule,
                                         StringRef contextPath) const {
  (void)contextPath;
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
    if (blockArg.getOwner() == currentModule.getBodyBlock()) {
      size_t argNum = blockArg.getArgNumber();
      return currentModule.getInputName(argNum).str();
    }
  }

  return "unnamed";
}

LogicalResult TimingGraph::build(const DelayModel *delayModel) {
  nodes.clear();
  arcs.clear();
  startPoints.clear();
  endPoints.clear();
  topoOrder.clear();
  reverseTopoOrder.clear();
  valueToNode.clear();

  // Use default model if none provided
  std::unique_ptr<DelayModel> defaultModel;
  if (!delayModel) {
    defaultModel = createDefaultDelayModel();
    delayModel = defaultModel.get();
  }
  delayModelName = delayModel->getName().str();

  LLVM_DEBUG(llvm::dbgs() << "Building timing graph for module: "
                          << module.getModuleName() << "\n");

  LogicalResult result = hierarchical ? buildHierarchicalGraph(*delayModel)
                                      : buildFlatGraph(*delayModel);
  if (failed(result))
    return failure();

  computeTopologicalOrder();

  LLVM_DEBUG(llvm::dbgs() << "Timing graph built: " << nodes.size()
                          << " nodes, " << arcs.size() << " arcs, "
                          << startPoints.size() << " start points, "
                          << endPoints.size() << " end points\n");

  return success();
}

LogicalResult TimingGraph::processOperation(Operation *op,
                                            const DelayModel &model,
                                            hw::HWModuleOp currentModule,
                                            StringRef contextPath,
                                            bool topContext) {
  // Skip constants
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();

  // Handle output operation - create end points for output ports
  if (auto outputOp = dyn_cast<hw::OutputOp>(op)) {
    if (!topContext)
      return success();
    for (auto [idx, operand] : llvm::enumerate(outputOp.getOperands())) {
      size_t width = getBitWidth(operand);
      for (size_t bit = 0; bit < width; ++bit) {
        // Create output port node
        std::string name = currentModule.getOutputName(idx).str();
        TimingNodeId id = createNode(operand, bit, TimingNodeKind::OutputPort,
                                     name, contextPath,
                                     /*addToLookup=*/false);
        auto *node = getNode(id);

        // Create arc from operand to output.
        // Use getOrCreateNode to handle graph-region use-before-def values.
        auto *fromNode = getOrCreateNode(operand, bit, currentModule,
                                         contextPath, topContext);
        createArc(fromNode, node, 0);
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
        getOrCreateNode(result, bit, currentModule, contextPath, topContext);
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
        TimingNodeId id = createNode(input, bit, TimingNodeKind::RegisterInput,
                                     name, contextPath,
                                     /*addToLookup=*/false);
        auto *node = getNode(id);

        // Use getOrCreateNode to handle graph-region use-before-def values.
        auto *fromNode =
            getOrCreateNode(input, bit, currentModule, contextPath, topContext);
        createArc(fromNode, node, 0);
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
        TimingNodeId id = createNode(input, bit, TimingNodeKind::RegisterInput,
                                     name, contextPath,
                                     /*addToLookup=*/false);
        auto *node = getNode(id);

        // Use getOrCreateNode to handle graph-region use-before-def values.
        auto *fromNode =
            getOrCreateNode(input, bit, currentModule, contextPath, topContext);
        createArc(fromNode, node, 0);
      }
    }
    return success();
  }

  // Handle combinational operations
  DelayContext ctx;
  ctx.op = op;
  int64_t delay = model.computeDelay(ctx).delay;

  for (auto result : op->getResults()) {
    size_t resultWidth = getBitWidth(result);
    for (size_t bit = 0; bit < resultWidth; ++bit) {
      auto *toNode =
          getOrCreateNode(result, bit, currentModule, contextPath, topContext);

      // Create arcs from operands
      for (auto operand : op->getOperands()) {
        size_t operandWidth = getBitWidth(operand);
        // For simplicity, connect all operand bits to result bits
        // More sophisticated handling could be added for extract/concat
        size_t srcBit = std::min(bit, operandWidth - 1);
        auto *fromNode = getOrCreateNode(operand, srcBit, currentModule,
                                         contextPath, topContext);
        createArc(fromNode, toNode, delay);
      }
    }
  }

  return success();
}

LogicalResult TimingGraph::buildFlatGraph(const DelayModel &model) {
  // Process input ports.
  for (auto arg : module.getBodyBlock()->getArguments()) {
    size_t width = getBitWidth(arg);
    for (size_t bit = 0; bit < width; ++bit)
      getOrCreateNode(arg, bit, module, "", /*topContext=*/true);
  }

  // Process operations.
  for (auto &op : module.getBodyBlock()->getOperations())
    if (failed(processOperation(&op, model, module, "", /*topContext=*/true)))
      return failure();

  return success();
}

LogicalResult TimingGraph::buildHierarchicalGraph(const DelayModel &model) {
  if (!circuit) {
    module.emitError("hierarchical timing graph requires parent module op");
    return failure();
  }
  llvm::SmallVector<StringAttr> stack;
  return buildModuleInContext(model, module, "", stack, /*topContext=*/true);
}

LogicalResult TimingGraph::buildModuleInContext(
    const DelayModel &model, hw::HWModuleOp currentModule,
    StringRef contextPath, llvm::SmallVectorImpl<StringAttr> &stack,
    bool topContext) {
  auto moduleName = currentModule.getModuleNameAttr();
  if (llvm::is_contained(stack, moduleName)) {
    currentModule.emitError("recursive module hierarchy is not supported by "
                            "timing analysis");
    return failure();
  }

  stack.push_back(moduleName);

  for (auto arg : currentModule.getBodyBlock()->getArguments()) {
    size_t width = getBitWidth(arg);
    for (size_t bit = 0; bit < width; ++bit)
      getOrCreateNode(arg, bit, currentModule, contextPath, topContext);
  }

  for (auto &op : currentModule.getBodyBlock()->getOperations()) {
    if (auto inst = dyn_cast<hw::InstanceOp>(op)) {
      auto childName = inst.getReferencedModuleNameAttr();
      auto childModule = circuit.lookupSymbol<hw::HWModuleOp>(childName);
      if (!childModule) {
        // Fall back to flat black-box behavior for unknown/external modules.
        if (failed(processOperation(&op, model, currentModule, contextPath,
                                    topContext))) {
          stack.pop_back();
          return failure();
        }
        continue;
      }

      std::string childContext =
          makeChildContext(contextPath, inst.getInstanceName());

      // Connect parent operands to child inputs.
      auto childArgs = childModule.getBodyBlock()->getArguments();
      for (auto [idx, operand] : llvm::enumerate(inst.getOperands())) {
        if (idx >= childArgs.size())
          continue;
        auto childArg = childArgs[idx];
        size_t width = std::min(getBitWidth(operand), getBitWidth(childArg));
        for (size_t bit = 0; bit < width; ++bit) {
          auto *from = getOrCreateNode(operand, bit, currentModule, contextPath,
                                       topContext);
          auto *to = getOrCreateNode(childArg, bit, childModule, childContext,
                                     /*topContext=*/false);
          createArc(from, to, 0);
        }
      }

      // Build child internals in this instance context.
      if (failed(buildModuleInContext(model, childModule, childContext, stack,
                                      /*topContext=*/false))) {
        stack.pop_back();
        return failure();
      }

      // Connect child outputs to parent instance results.
      auto outputOp =
          dyn_cast<hw::OutputOp>(childModule.getBodyBlock()->getTerminator());
      if (!outputOp)
        continue;
      for (auto [idx, instResult] : llvm::enumerate(inst.getResults())) {
        if (idx >= outputOp.getNumOperands())
          continue;
        Value childOutput = outputOp.getOperand(idx);
        size_t width =
            std::min(getBitWidth(childOutput), getBitWidth(instResult));
        for (size_t bit = 0; bit < width; ++bit) {
          auto *from = getOrCreateNode(childOutput, bit, childModule,
                                       childContext, /*topContext=*/false);
          auto *to = getOrCreateNode(instResult, bit, currentModule,
                                     contextPath, topContext);
          createArc(from, to, 0);
        }
      }
      continue;
    }

    if (failed(processOperation(&op, model, currentModule, contextPath,
                                topContext))) {
      stack.pop_back();
      return failure();
    }
  }

  stack.pop_back();
  return success();
}

void TimingGraph::computeTopologicalOrder() {
  // Kahn's algorithm for correct topological sort
  topoOrder.clear();
  reverseTopoOrder.clear();

  // Compute in-degree for each node
  llvm::DenseMap<TimingNode *, unsigned> inDegree;
  for (const auto &node : nodes)
    inDegree[node.get()] = node->getFanin().size();

  // Seed queue with zero-in-degree nodes
  llvm::SmallVector<TimingNode *, 64> queue;
  for (const auto &node : nodes) {
    if (inDegree[node.get()] == 0)
      queue.push_back(node.get());
  }

  while (!queue.empty()) {
    auto *node = queue.pop_back_val();
    topoOrder.push_back(node);

    for (auto *arc : node->getFanout()) {
      auto *succ = arc->getTo();
      unsigned &deg = inDegree[succ];
      assert(deg > 0 && "in-degree underflow");
      if (--deg == 0)
        queue.push_back(succ);
    }
  }

  // Build reverse topological order
  reverseTopoOrder.assign(topoOrder.rbegin(), topoOrder.rend());

  LLVM_DEBUG(llvm::dbgs() << "Topological order: " << topoOrder.size()
                          << " nodes (out of " << nodes.size() << ")\n");
}
