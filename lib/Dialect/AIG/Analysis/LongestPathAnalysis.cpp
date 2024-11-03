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
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/JSON.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "aig-longest-path-analysis"
using namespace circt;
using namespace aig;

static size_t getBitWidth(Value value) {
  if (auto vecType = value.getType().dyn_cast<seq::ClockType>())
    return 1;
  return value.getType().getIntOrFloatBitWidth();
}

static bool isRootValue(Value value) {
  if (auto arg = value.dyn_cast<BlockArgument>())
    return true;
  return isa<seq::CompRegOp, seq::FirRegOp, hw::InstanceOp, seq::FirMemReadOp>(
      value.getDefiningOp());
}

Value createOrReuseExtract(OpBuilder &rewriter, Location loc, Value operand,
                           size_t lowBit, size_t width) {
  if (auto concatOp = operand.getDefiningOp<comb::ConcatOp>()) {
    SmallVector<Value> newOperands;
    for (auto operand : llvm::reverse(concatOp.getOperands())) {
      auto opWidth = operand.getType().getIntOrFloatBitWidth();
      if (width == 0)
        break;

      if (lowBit >= opWidth) {
        lowBit -= opWidth;
        continue;
      }

      if (lowBit == 0 && width == opWidth) {
        newOperands.push_back(operand);
        break;
      }

      // lowBit < width
      size_t extractWidth = std::min(opWidth - lowBit, width);
      newOperands.push_back(
          createOrReuseExtract(rewriter, loc, operand, lowBit, extractWidth));
      width -= extractWidth;
      lowBit = 0;
    }
    if (newOperands.size() == 1)
      return newOperands.front();

    std::reverse(newOperands.begin(), newOperands.end());
    return rewriter.create<comb::ConcatOp>(loc, newOperands);
  }

  return rewriter.createOrFold<comb::ExtractOp>(
      loc, rewriter.getIntegerType(width), operand, lowBit);
}
struct ShrinkDelayPattern : OpRewritePattern<aig::DelayOp> {
  using OpRewritePattern<aig::DelayOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DelayOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> newDelays;
    SmallVector<Value> newOperands;
    DenseMap<std::tuple<Value, size_t, size_t>, size_t> operandToIdx;
    SmallVector<std::tuple<Value, int64_t, size_t>> worklist;
    DenseMap<Value, int64_t> observedMaxDelay;
    // if (!llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
    //       return isa<seq::FirRegOp, hw::OutputOp, hw::InstanceOp>(user);
    //     }))
    //   return failure();

    bool changed = false;
    for (auto [operand, delay] :
         llvm::reverse(llvm::zip(op.getOperands(), op.getDelays())))
      worklist.push_back({operand, delay, 0});

    while (!worklist.empty()) {
      auto [operand, delay, depth] = worklist.pop_back_val();

      if (observedMaxDelay[operand] > delay) {
        continue;
      }

      observedMaxDelay[operand] = delay;

      if (auto delayOp = operand.getDefiningOp<DelayOp>()) {
        for (auto [dOpOperand, dOpDelay] : llvm::reverse(
                 llvm::zip(delayOp.getOperands(), delayOp.getDelays()))) {
          if (observedMaxDelay[dOpOperand] > dOpDelay + delay) {
            continue;
          }
          observedMaxDelay[dOpOperand] = dOpDelay + delay;
          worklist.push_back({dOpOperand, dOpDelay + delay, depth + 1});
          changed = true;
        }
      } else if (auto constantOp = operand.getDefiningOp<hw::ConstantOp>()) {
        // Delay of constant is 0.
        changed = true;
      } else if (auto replicateOp =
                     operand.getDefiningOp<comb::ReplicateOp>()) {
        worklist.push_back({replicateOp.getInput(), delay, depth + 1});
        changed = true;
      } else {
        std::tuple<Value, size_t, size_t> operandTuple;
        if (auto extractOp = operand.getDefiningOp<comb::ExtractOp>()) {
          operandTuple =
              std::make_tuple(extractOp.getInput(), extractOp.getLowBit(),
                              extractOp.getType().getIntOrFloatBitWidth());
        } else {
          operandTuple = std::make_tuple(
              operand, 0, operand.getType().getIntOrFloatBitWidth());
        }

        if (operandToIdx.count(operandTuple)) {
          newDelays[operandToIdx[operandTuple]] =
              std::max(newDelays[operandToIdx[operandTuple]], delay);
          changed = true;
        } else {
          operandToIdx[operandTuple] = newOperands.size();
          newDelays.push_back(delay);
          if (std::get<1>(operandTuple) == 0 &&
              std::get<2>(operandTuple) ==
                  operand.getType().getIntOrFloatBitWidth()) {
            newOperands.push_back(operand);
          } else {
            if (operand.getType() != op.getType()) {
              auto replicateOp = rewriter.create<comb::ReplicateOp>(
                  op.getLoc(), op.getType(), operand);
              operand = replicateOp.getResult();
            }
            newOperands.push_back(createOrReuseExtract(
                rewriter, op.getLoc(), operand, std::get<1>(operandTuple),
                std::get<2>(operandTuple)));
          }
        }
      }
    }

    // Nothing changed.
    if (!changed)
      return failure();

    if (newOperands.empty()) {
      // Everything is constant.
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    rewriter.replaceOpWithNewOp<aig::DelayOp>(
        op, op.getType(), newOperands,
        rewriter.getDenseI64ArrayAttr(newDelays));
    return success();
  }
};

// LongestPathAnalysis::LongestPathAnalysis(Operation *moduleOp,
//                                              mlir::AnalysisManager &am)
//     : instanceGraph(&am.getAnalysis<igraph::InstanceGraph>()) {}
//
// LongestPathAnalysis::ModuleInfo *
// LongestPathAnalysis::getModuleInfo(hw::HWModuleOp module) {
//   {
//     auto it = moduleInfoCache.find(module.getModuleNameAttr());
//     if (it != moduleInfoCache.end())
//       return it->second.get();
//   }
//   auto *topNode = instanceGraph->lookup(module.getModuleNameAttr());
// }

/*
static llvm::json::Object
getModuleResourceUsageJSON(const ResourceUsageAnalysis::ResourceUsage &usage) {
  llvm::json::Object obj;
  obj["numAndInverterGates"] = usage.getNumAndInverterGates();
  obj["numDFFBits"] = usage.getNumDFFBits();
  return obj;
}

// This creates a fully-elaborated information but should be ok for now.
static llvm::json::Object getModuleResourceUsageJSON(
    const ResourceUsageAnalysis::ModuleResourceUsage &usage) {
  llvm::json::Object obj;
  obj["local"] = getModuleResourceUsageJSON(usage.local);
  obj["total"] = getModuleResourceUsageJSON(usage.total);
  obj["moduleName"] = usage.moduleName.getValue();
  SmallVector<llvm::json::Value> instances;
  for (const auto &instance : usage.instances) {
    llvm::json::Object child;
    child["instanceName"] = instance.instanceName.getValue();
    child["moduleName"] = instance.moduleName.getValue();
    child["usage"] = getModuleResourceUsageJSON(*instance.usage);
    instances.push_back(std::move(child));
  }
  obj["instances"] = llvm::json::Array(std::move(instances));
  return obj;
}

void ResourceUsageAnalysis::ModuleResourceUsage::emitJSON(
    raw_ostream &os) const {
  os << getModuleResourceUsageJSON(*this);
}

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTRESOURCEUSAGEANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

namespace {
struct PrintResourceUsageAnalysisPass
    : public impl::PrintResourceUsageAnalysisBase<
          PrintResourceUsageAnalysisPass> {
  using PrintResourceUsageAnalysisBase::PrintResourceUsageAnalysisBase;

  using PrintResourceUsageAnalysisBase::printSummary;
  using PrintResourceUsageAnalysisBase::outputJSONFile;
  using PrintResourceUsageAnalysisBase::topModuleName;
  void runOnOperation() override;
};
} // namespace

void PrintResourceUsageAnalysisPass::runOnOperation() {
  auto mod = getOperation();
  if (topModuleName.empty()) {
    mod.emitError()
        << "'top-name' option is required for PrintResourceUsageAnalysis";
    return signalPassFailure();
  }
  auto &symTbl = getAnalysis<mlir::SymbolTable>();
  auto top = symTbl.lookup<hw::HWModuleOp>(topModuleName);
  if (!top) {
    mod.emitError() << "top module '" << topModuleName << "' not found";
    return signalPassFailure();
  }
  auto &resourceUsageAnalysis = getAnalysis<ResourceUsageAnalysis>();
  auto usage = resourceUsageAnalysis.getResourceUsage(top);

  if (printSummary) {
    llvm::errs() << "// ------ ResourceUsageAnalysis Summary -----\n";
    llvm::errs() << "Top module: " << topModuleName << "\n";
    llvm::errs() << "Total number of and-inverter gates: "
                 << usage->getTotal().getNumAndInverterGates() << "\n";
    llvm::errs() << "Total number of DFF bits: "
                 << usage->getTotal().getNumDFFBits() << "\n";

  }
  if (!outputJSONFile.empty()) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputJSONFile, ec);
    if (ec) {
      emitError(UnknownLoc::get(&getContext()))
          << "failed to open output JSON file '" << outputJSONFile << "': "
          << ec.message();
      return signalPassFailure();
    }
    usage->emitJSON(os);
  }
}
*/

namespace circt {
namespace aig {
#define GEN_PASS_DEF_PRINTLONGESTPATHANALYSIS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

struct LocalPathAnalysisTransform {
  struct BitRange {
    Value operand;
    size_t lowBit;
    size_t width;
  };
  SetVector<comb::ExtractOp> extractOps;
  SetVector<comb::ConcatOp> concatOps;
  SetVector<comb::ReplicateOp> replicateOps;
  SetVector<aig::DelayOp> delayOps;
  SmallVector<Operation *> pendingOps;

  // DenseMap<BitRange, Value> bitRangeToValue;
  Value getOrCreateBitRange(OpBuilder &builder, Value operand, size_t lowBit,
                            size_t width);
  LogicalResult lower(OpBuilder &builder, aig::AndInverterOp op);
  LogicalResult splitDelayOp(OpBuilder &builder, aig::DelayOp delayOp,
                             ArrayRef<size_t> splitPos,
                             ArrayRef<int64_t> newDelays,
                             ArrayRef<Value> newOperands);

  template <typename... Args>
  Value createReplicate(OpBuilder &builder, Args &&...args);
  template <typename... Args>
  Value createConcat(OpBuilder &builder, Args &&...args);
  template <typename... Args>
  Value createExtract(OpBuilder &builder, Args &&...args);
  template <typename... Args>
  Value createDelay(OpBuilder &builder, Args &&...args);

  LogicalResult run(hw::HWModuleOp mod);
  LogicalResult buildGraph(hw::HWModuleOp mod);

  Value createOrReuseExtract2(OpBuilder &rewriter, Location loc, Value operand,
                              size_t lowBit, size_t width) {
    if (operand.getType().getIntOrFloatBitWidth() < lowBit + width)
      llvm::dbgs() << "createOrReuseExtract: " << operand << " " << lowBit
                   << " " << width << "\n";
    if (auto concatOp = operand.getDefiningOp<comb::ConcatOp>()) {
      SmallVector<Value> newOperands;
      for (auto operand : llvm::reverse(concatOp.getOperands())) {
        auto opWidth = operand.getType().getIntOrFloatBitWidth();
        if (width == 0)
          break;

        if (lowBit >= opWidth) {
          lowBit -= opWidth;
          continue;
        }

        if (lowBit == 0 && width == opWidth) {
          newOperands.push_back(operand);
          break;
        }

        // lowBit < width
        size_t extractWidth = std::min(opWidth - lowBit, width);
        newOperands.push_back(
            createExtract(rewriter, loc, operand, lowBit, extractWidth));
        width -= extractWidth;
        lowBit = 0;
      }
      if (newOperands.size() == 1)
        return newOperands.front();

      std::reverse(newOperands.begin(), newOperands.end());
      return createConcat(rewriter, loc, newOperands);
    }

    return createExtract(rewriter, loc, operand, lowBit, width);
  }
};

template <typename... Args>
Value LocalPathAnalysisTransform::createReplicate(OpBuilder &builder,
                                                  Args &&...args) {
  auto newReplicateOp =
      builder.createOrFold<comb::ReplicateOp>(std::forward<Args>(args)...);
  if (auto replicateOp =
          newReplicateOp.template getDefiningOp<comb::ReplicateOp>())
    replicateOps.insert(replicateOp);
  return newReplicateOp;
}

template <typename... Args>
Value LocalPathAnalysisTransform::createDelay(OpBuilder &builder,
                                              Args &&...args) {
  auto newDelayOp =
      builder.createOrFold<aig::DelayOp>(std::forward<Args>(args)...);
  if (auto delayOp = newDelayOp.template getDefiningOp<aig::DelayOp>())
    delayOps.insert(delayOp);
  return newDelayOp;
}

template <typename... Args>
Value LocalPathAnalysisTransform::createConcat(OpBuilder &builder,
                                               Args &&...args) {
  auto newConcatOp =
      builder.createOrFold<comb::ConcatOp>(std::forward<Args>(args)...);
  if (auto concatOp = newConcatOp.template getDefiningOp<comb::ConcatOp>())
    concatOps.insert(concatOp);
  return newConcatOp;
}

template <typename... Args>
Value LocalPathAnalysisTransform::createExtract(OpBuilder &builder,
                                                Args &&...args) {
  auto newExtractOp =
      builder.createOrFold<comb::ExtractOp>(std::forward<Args>(args)...);
  if (auto extractOp = newExtractOp.template getDefiningOp<comb::ExtractOp>())
    extractOps.insert(extractOp);
  return newExtractOp;
}

LogicalResult LocalPathAnalysisTransform::splitDelayOp(
    OpBuilder &builder, aig::DelayOp op, ArrayRef<size_t> splitPos,
    ArrayRef<int64_t> newDelays, ArrayRef<Value> newOperandValues) {
  // Delay(Delay(d), Delay(e), Concat(a, b, c)) -> Concat(Dealy(e), Delay(a),
  // Delay(b), Delay(c))
  if (splitPos.size() <= 1) {
    auto value = createDelay(builder, op.getLoc(), op.getType(),
                             newOperandValues, newDelays);
    op.getResult().replaceAllUsesWith(value);
    return success();
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(newOperands.size());
  SmallVector<Value> results;
  results.reserve(splitPos.size() - 1);
  // size_t bitPos = 0;
  // size_t width = splitPos[0];
  // llvm::dbgs() << "splitDelayOp: " << op << " " << splitPos.size() << " "
  //              << bitPos << " " << width << "\n";
  // for (auto splitPos : splitPos)
  //   llvm::dbgs() << "splitPos: " << splitPos << "\n";
  for (auto i = 0; i < splitPos.size(); ++i) {
    newOperands.resize(0);
    auto bitPos = i == 0 ? 0 : splitPos[i - 1];
    auto width = splitPos[i] - bitPos;
    // llvm::dbgs() << "width: " << width << " " << bitPos << "\n";
    for (auto operand : newOperandValues) {
      if (operand.getType().isInteger(1)) {
        newOperands.push_back(operand);
      } else {
        if (operand.getType() != op.getType())
          operand =
              createReplicate(builder, op.getLoc(), op.getType(), operand);
        newOperands.push_back(createOrReuseExtract2(builder, op.getLoc(),
                                                    operand, bitPos, width));
      }
    }
    results.push_back(createDelay(builder, op.getLoc(),
                                  builder.getIntegerType(width), newOperands,
                                  newDelays));
  }

  // for (auto operand : llvm::reverse(newOperands)) {
  //   auto bitWidth = operand.getType().getIntOrFloatBitWidth();
  //   newOperands.push_back(operand);
  //   // FIXME: This has to a bug!
  //   for (auto rest : op.getOperands()) {
  //     if (rest == concatOp)
  //       continue;
  //     // FIXME: Reuse extracted op if possible.
  //     newOperands.push_back(
  //         createOrReuseExtract(builder, op.getLoc(), rest, bitPos,
  //         bitWidth));
  //   }
  //   newOperands.push_back(operand);
  //   bitPos += bitWidth;
  //   auto result = builder.createOrFold<aig::DelayOp>(
  //       op.getLoc(), operand.getType(), newOperands, op.getDelays());
  //   results.push_back(result);
  //   newOperands.resize(concatIdx);
  // }

  std::reverse(results.begin(), results.end());

  op.getResult().replaceAllUsesWith(
      createConcat(builder, op.getLoc(), op.getType(), results));
  return success();
}

struct Node {
  enum Kind {
    Input,
    Delay,
    Output,
    Concat,
    Replicate,
    Extract,
    Constant,
  };
  Kind kind;
  size_t width;
  Node(Kind kind, size_t width) : kind(kind), width(width) {}
  size_t getWidth() const { return width; }
  Kind getKind() const { return kind; }

  virtual Node *query(size_t bitOffset) = 0;
  virtual ~Node() {}
  virtual void dump() const = 0;
};

struct ConstantNode : Node {
  ConstantNode(size_t width, ConstantNode *boolNode)
      : Node(Kind::Constant, width), boolNode(boolNode) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Constant; }
  void dump() const override {
    llvm::dbgs() << "(constant node " << width << ")";
  }
  Node *query(size_t bitOffset) override { return boolNode; }

private:
  ConstantNode *boolNode = nullptr;
};

struct InputNode : Node {
  Value value; // port, instance output, or register.
  size_t bitPos;
  InputNode(Value value, size_t bitPos)
      : Node(Kind::Input, 1), value(value), bitPos(bitPos) {}

  static bool classof(const Node *e) { return e->getKind() == Kind::Input; }
  void setBitPos(size_t bitPos) { this->bitPos = bitPos; }
  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "input node has no bit offset");
    return this;
  }
  void dump() const override {
    llvm::dbgs() << "(input node " << value << " " << bitPos << ")\n";
  }
  // ~InputNode() {
  //   llvm::dbgs() << "destroy input node " << value << "\n";
  // }
};

struct DelayNode : Node {

  DelayNode &operator=(const DelayNode &other) = default;
  DelayNode(const DelayNode &other) = default;
  DelayNode(Value value, size_t bitPos)
      : Node(Kind::Delay, 1), value(value), bitPos(bitPos), edges(),
        computedResult() {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Delay; }

  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "delay node has no bit offset");
    return this;
  }
  void addEdge(Node *node, int64_t delay) {
    edges.push_back(std::make_pair(delay, node));
  }
  void setBitPos(size_t bitPos) { this->bitPos = bitPos; }
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
    if (!computedResult)
      computeResult();

    assert(computedResult && "result not computed");
    return computedResult.value();
  }

  std::optional<SmallVector<std::pair<int64_t, InputNode *>>> computedResult;

  ~DelayNode() override {
    // llvm::dbgs() << "destroy delay node " << value << " " << this << "\n";
    edges.clear();
    computedResult.reset();
    // llvm::dbgs() << "done";
  }

  void dump() const override {
    llvm::dbgs() << "(delay node " << value << " " << bitPos << "\n";
    for (auto edge : edges) {
      llvm::dbgs() << "edge: delay " << edge.first;
      edge.second->dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << ")\n";
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
  OutputNode(Operation *op, size_t operandIdx, size_t bitPos)
      : Node(Kind::Output, 1), op(op), operandIdx(operandIdx), bitPos(bitPos) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Output; }
  Node *query(size_t bitOffset) override {
    assert(bitOffset == 0 && "delay node has no bit offset");
    return this;
  }
  ~OutputNode() = default;
};

struct ConcatNode : Node {
  Value value;
  SmallVector<Node *> nodes;
  ConcatNode(Value value, ArrayRef<Node *> nodes)
      : Node(Kind::Concat, getBitWidth(value)), value(value), nodes(nodes) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Concat; }
  ~ConcatNode() override { nodes.clear(); }
  ConcatNode(const ConcatNode &other) = default;
  ConcatNode &operator=(const ConcatNode &other) = default;
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
  void dump() const override {
    llvm::dbgs() << "(concat node " << value << "\n";
    for (auto node : nodes) {
      node->dump();
    }
    llvm::dbgs() << ")\n";
  }
};

struct ReplicateNode : Node {
  Value value;
  Node *node;
  ReplicateNode(Value value, Node *node)
      : Node(Kind::Replicate, getBitWidth(value)), value(value), node(node) {
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
  void dump() const override {
    llvm::dbgs() << "(replicate node " << value << "\n";
    node->dump();
    llvm::dbgs() << ")\n";
  }
};

struct ExtractNode : Node {
  Value value;
  size_t lowBit;
  Node *input;
  ExtractNode(const ExtractNode &other) = default;
  ExtractNode &operator=(const ExtractNode &other) = default;
  ExtractNode(Value value, size_t lowBit, Node *input)
      : Node(Kind::Extract, getBitWidth(value)), value(value), lowBit(lowBit),
        input(input) {}
  static bool classof(const Node *e) { return e->getKind() == Kind::Extract; }
  ~ExtractNode() = default;

  Node *query(size_t bitOffset) override {
    // exract(a, 2) : i4 -> query(2)
    return input->query(bitOffset + lowBit);
  }
  void dump() const override {
    llvm::dbgs() << "(extract node " << value << " " << lowBit << "\n";
    input->dump();
    llvm::dbgs() << ")\n";
  }
};

struct Graph {
  DenseMap<Value, Node *> valueToNodes;
  DenseMap<std::tuple<Operation *, size_t, size_t>, OutputNode *> outputNodes;
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

  ConstantNode dummy{0, nullptr};
  void addConstantNode(Value value) {
    auto *nodePtr = allocateNode<ConstantNode>(
        value.getType().getIntOrFloatBitWidth(), &dummy);
    setResultValue2(value, nodePtr);
  }

  // FIXME: SpecificBumpPtrAllocator crashes for some reason. For now use unique
  // ptrs.

  template <typename NodeTy, typename... Args>
  NodeTy *allocateNode(Args &&...args) {
    auto nodePtr = std::make_unique<NodeTy>(std::forward<Args>(args)...);
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
        if (valueToNodes.find(operand) == valueToNodes.end())
          llvm::errs() << "operand: " << operand << "\n";
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

  LogicalResult addConcatNode(Value value, mlir::OperandRange inputs) {
    size_t width = value.getType().getIntOrFloatBitWidth();
    SmallVector<Node *> nodes;
    for (auto operand : llvm::reverse(inputs)) {
      auto *inputNode = valueToNodes[operand];
      // llvm::dbgs() << "inputNode: " << operand << "\n";
      assert(inputNode && "input node not found");
      nodes.push_back(inputNode);
    }

    setResultValue(value, nodes);
    return success();
  }

  LogicalResult addExtractNode(Value value, Value operand, size_t lowBit) {
    auto *inputNode = valueToNodes[operand];
    assert(inputNode && "input node not found");
    // inputNode->dump();
    auto *node = allocateNode<ExtractNode>(value, lowBit, inputNode);
    assert(node);

    setResultValue2(value, node);
    return success();
  }

  LogicalResult addReplicateNode(Value value, Value operand) {
    auto *inputNode = valueToNodes[operand];
    assert(inputNode && "input node not found");
    auto *node = allocateNode<ReplicateNode>(value, inputNode);
    assert(node && "node allocation failed");
    setResultValue2(value, node);
    return success();
  }
};

LogicalResult LocalPathAnalysisTransform::buildGraph(hw::HWModuleOp mod) {
  // mod.dump();
  Graph graph;
  for (auto arg : mod.getBodyBlock()->getArguments()) {
    graph.addInputNode(arg);
  }

  SmallVector<Operation *> interestingOps;
  // Add input nodes.
  mod.walk([&](Operation *op) {
    if (isa<seq::FirRegOp, seq::CompRegOp, hw::InstanceOp, seq::FirMemReadOp>(
        op)) {
      for (auto result : op->getResults())
        graph.addInputNode(result);
    } else if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp,
                   hw::ConstantOp, aig::AndInverterOp>(op)) {
      interestingOps.push_back(op);
    }
  });

  auto isOperandReady = [&](Value value, Operation *op) {
    return isRootValue(value);
  };
  mlir::computeTopologicalSorting(interestingOps, isOperandReady);

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
    } else if (auto constantOp = dyn_cast<hw::ConstantOp>(op)) {
      graph.addConstantNode(constantOp);
    }
  }

  // Add delay nodes.
  // mod.walk([&](Operation *op) {

  // });
  // llvm::dbgs() << "done\n";
  return success();
}

LogicalResult LocalPathAnalysisTransform::lower(OpBuilder &builder,
                                                aig::AndInverterOp op) {
  auto loc = op.getLoc();
  builder.setInsertionPoint(op);
  // Replace with DelayOp.
  // 1 -> 1, 2 -> 1, 3 -> 2, 4, -> 2
  uint64_t delay = std::max(1u, llvm::Log2_64_Ceil(op.getOperands().size()));
  SmallVector<int64_t> delays(op.getOperands().size(), delay);
  auto delaysAttr = builder.getDenseI64ArrayAttr(delays);
  auto delayOp =
      createDelay(builder, loc, op.getType(), op.getOperands(), delaysAttr);

  op.getResult().replaceAllUsesWith(delayOp);
  op.erase();
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

LogicalResult LocalPathAnalysisTransform::run(hw::HWModuleOp mod) {
  return buildGraph(mod);

  // NOTE: The result IR could be huge so instead of relying on
  // GreedyRewriteDriver and DialectConversion manually iterate over the
  // operations.

  OpBuilder builder(mod.getContext());
  // 1. Replace AndInverterOp with DelayOp.
  // 2. Delay(replicate(x)) -> Delay(x)
  // 3. Eliminate ConcatOp
  //   while(there is concat op) {
  //     // Check users.
  //     if (user is DelayOp) {
  //       need to split delay op. it may introduce extract op and concat.
  //     }
  //   }
  // 4. Eliminate existing ExtractOp
  //   while(there is extract op) {
  //     auto input = extract.getInput();
  //     auto delay = input.getDefiningOp<aig::DelayOp>();
  //     if(delay) {
  //       replace extract with delay.getOperands().back();
  //       worklist.push_back(input);
  //       continue;
  //     }
  //     auto concat = input.getDefiningOp<comb::ConcatOp>();
  //     These could introudce new concat ops. -- record it.
  //     if(concat) {
  //       replace extract with concat.getOperands().front();
  //       worklist.push_back(input);
  //       continue;
  //     }
  //     if(replicate) {
  //       replace extract with replicate.getInput();
  //       push_to_worklist if necessary
  //       worklist.push_back(input);
  //       continue;
  //     }
  //   }
  // Loop 3 and 4 until no more extract and concat ops.
  mod.walk([&](Operation *op) {
    if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
      if (failed(lower(builder, andInverterOp)))
        return WalkResult::interrupt();
    } else if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
      extractOps.insert(extractOp);
    } else if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
      concatOps.insert(concatOp);
    } else if (auto replicateOp = dyn_cast<comb::ReplicateOp>(op)) {
      replicateOps.insert(replicateOp);
    }
    return WalkResult::advance();
  });

  // mlir::PatternRewriter rewriter(mod.getContext());
  // mlir::DominanceInfo domInfo(mod);
  // mlir::eliminateCommonSubExpressions(rewriter, domInfo, mod);
  // ExtractReplicateConversion extractReplicateConversionPattern(
  //     mod.getContext());

  builder.setInsertionPointToStart(mod.getBodyBlock());

  size_t numConcatOpsVisited = 0;
  while (!extractOps.empty() || !delayOps.empty() || !concatOps.empty()) {
    /*
    while (!replicateOps.empty()) {
      auto replicateOp = replicateOps.pop_back_val();
      LLVM_DEBUG(llvm::dbgs() << "replicateOp: " << replicateOp << "\n");
      replicateOp->replaceUsesWithIf(
          ArrayRef<Value>(replicateOp.getInput()),
          [&](OpOperand &use) { return isa<aig::DelayOp>(use.getOwner()); });

      if (replicateOp.use_empty()) {
        replicateOp.erase();
        continue;
      }
      bool pending = false;

      for (auto user : replicateOp->getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<comb::ExtractOp>(
                [&](comb::ExtractOp extractOp) { pending = true; })
            .Case<comb::ConcatOp>(
                [&](comb::ConcatOp concatOp) { pending = true; })
            .Default([&](Operation *) {});
      }

      if (pending)
        pendingOps.push_back(replicateOp);
    }
    */
    while (!concatOps.empty()) {
      numConcatOpsVisited++;
      auto concatOp =
          concatOps.pop_back_val(); // Check users of ConcatOp and lower them.
      LLVM_DEBUG(llvm::dbgs()
                 << "concatOp: " << concatOp << " " << numConcatOpsVisited
                 << " " << concatOps.size() << "\n");
      if (concatOp.use_empty()) {
        concatOp.erase();
        continue;
      }

      for (auto user : llvm::make_early_inc_range(concatOp->getUsers())) {
        TypeSwitch<Operation *>(user)
            .Case<aig::DelayOp>(
                [&](aig::DelayOp delayOp) { delayOps.insert(delayOp); })
            .Case<comb::ExtractOp>([&](comb::ExtractOp extractOp) {
              // Introduce new extract op.
              extractOps.insert(extractOp);
            })
            .Default([&](Operation *) {
              // Introduce new concat op.
            });
      }
    }

    while (!extractOps.empty()) {
      auto extractOp = extractOps.pop_back_val();
      builder.setInsertionPoint(extractOp);
      auto input = extractOp.getInput();

      LLVM_DEBUG(llvm::dbgs() << "extractOp: " << extractOp << "\n");
      if (isRootValue(input)) {
        continue;
      }

      if (extractOp.use_empty()) {
        extractOp.erase();
        continue;
      }

      if (auto delayOp = input.getDefiningOp<aig::DelayOp>()) {
        SmallVector<Value> newOperands;
        for (auto operand : delayOp.getOperands()) {
          if (operand.getType() != delayOp.getType()) {
            if (operand.getType().isInteger(1)) {
              newOperands.push_back(operand);
              continue;
            } else {
              operand = builder.create<comb::ReplicateOp>(
                  extractOp.getLoc(), delayOp.getType(), operand);
            }
          }

          newOperands.push_back(createOrReuseExtract2(
              builder, extractOp.getLoc(), operand, extractOp.getLowBit(),
              extractOp.getType().getIntOrFloatBitWidth()));
        }
        auto newDelayOp =
            createDelay(builder, extractOp.getLoc(), extractOp.getType(),
                        newOperands, delayOp.getDelays());
        extractOp.replaceAllUsesWith(newDelayOp);
        // Split delay op.
      } else if (auto concatOp = input.getDefiningOp<comb::ConcatOp>()) {
        Value newOp = createOrReuseExtract2(
            builder, extractOp.getLoc(), concatOp, extractOp.getLowBit(),
            extractOp.getType().getIntOrFloatBitWidth());
        extractOp.replaceAllUsesWith(newOp);
        // Introduce new concat op.
      } else if (auto replicateOp = input.getDefiningOp<comb::ReplicateOp>()) {
        // Introduce new replicate op.
      } else if (auto constantOp = input.getDefiningOp<hw::ConstantOp>()) {
        // Delay of constant is 0.
      }
    }
    while (!delayOps.empty()) {
      auto delayOp = delayOps.pop_back_val();
      LLVM_DEBUG(llvm::dbgs() << "delayOp: " << delayOp << "\n");
      builder.setInsertionPoint(delayOp);
      // Split delay op.
      SetVector<size_t> splitPos;
      SmallVector<Value> newOperands;
      DenseMap<Value, size_t> opToIdx;
      SmallVector<int64_t> delays;
      bool changed = false;
      auto addToNewOperands = [&](Value op, int64_t delay) {
        if (!opToIdx.contains(op)) {
          newOperands.push_back(op);
          delays.push_back(delay);
          opToIdx[op] = newOperands.size() - 1;
        } else {
          auto idx = opToIdx[op];
          delays[idx] = std::max(delays[idx], delay);
        }
      };
      for (auto [operand, delay] :
           llvm::zip(delayOp.getInputs(), delayOp.getDelays())) {
        if (auto concatOp = operand.getDefiningOp<comb::ConcatOp>()) {
          size_t bitPos = 0;
          for (auto concatOperand : llvm::reverse(concatOp.getOperands())) {
            bitPos += concatOperand.getType().getIntOrFloatBitWidth();
            splitPos.insert(bitPos);
          }

          changed = concatOp.getOperands().size() > 1;
          addToNewOperands(concatOp, delay);

        } else if (auto constantOp = operand.getDefiningOp<hw::ConstantOp>()) {
          // Delay of constant is 0.
          changed = true;
          continue;
        } else if (auto replicateOp =
                       operand.getDefiningOp<comb::ReplicateOp>()) {
          changed = true;
          addToNewOperands(replicateOp.getInput(), delay);
        } else if (auto newDelayOp = operand.getDefiningOp<aig::DelayOp>()) {
          for (auto [newOperand, newDelay] :
               llvm::zip(newDelayOp.getInputs(), newDelayOp.getDelays())) {
            addToNewOperands(newOperand, delay + newDelay);
          }

          changed = true;
        } else {
          addToNewOperands(operand, delay);
        }
      }
      if (!changed)
        continue;
      auto vector = splitPos.takeVector();
      std::sort(vector.begin(), vector.end());
      splitDelayOp(builder, delayOp, vector, delays, newOperands);
    }
  }
  mlir::PatternRewriter rewriter(mod.getContext());
  // mlir::DominanceInfo domInfo(mod);
  // mlir::eliminateCommonSubExpressions(rewriter, domInfo, mod);
  // RewritePatternSet patterns(mod.getContext());
  // patterns.add<ShrinkDelayPattern>(mod.getContext());

  ShrinkDelayPattern shrinkDelayPattern(mod.getContext());

  // mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  // mlir::GreedyRewriteConfig config;
  // config.useTopDownTraversal = true;
  SetVector<Operation *> finalize;
  mod.walk([&](Operation *op) {
    if (op->getNumResults() == 1 && isRootValue(op->getResult(0))) {
      finalize.insert(op);
    }
    if (isa<hw::OutputOp, hw::InstanceOp>(op)) {
      finalize.insert(op);
    }
    return WalkResult::advance();
  });

  // while (!finalize.empty()) {
  //   auto op = finalize.pop_back_val();
  //   if (auto delayOp = dyn_cast<aig::DelayOp>(op)) {
  //     // llvm::errs() << "delayOp: " << delayOp << "\n";
  //     rewriter.setInsertionPoint(delayOp);
  //     shrinkDelayPattern.matchAndRewrite(delayOp, rewriter);
  //     continue;
  //   }
  //   for (auto operand : op->getOperands())
  //     if (auto operandOp = operand.getDefiningOp())
  //       finalize.insert(operandOp);
  // }
  // mlir::applyPatternsAndFoldGreedily(mod, frozen, config);

  // AndInverterToDelayConversion aigToDelay(getContext());
  // mlir::GreedyRewriteConfig config;
  // config.useTopDownTraversal = true;
  // if (failed(mlir::applyPatternsAndFoldGreedily(mod, frozen, config))) {
  //   llvm::errs() << "Failed to apply patterns and fold greedily\n";
  //   return failure();
  // }

  // mod.walk([&](Operation *op) {
  //   // 2. Hoist Concat and Sink ExtractOp.
  //   if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
  //     if (isRootValue(extractOp.getInput())) {
  //       legalizedValues.insert(extractOp.getInput());
  //       return WalkResult::advance();
  //     }

  //     // ExtractOp(AndDelay(a, b, c)) -> AndDelay(Extract(b))
  //     if (auto delayOp =
  //     extractOp.getInput().getDefiningOp<aig::DelayOp>())
  //     {
  //       auto newOperands = delayOp.getOperands();
  //       newOperands[extractOp.getLowBit() / delayOp.getDelays()[0]] =
  //           extractOp.getInput();
  //       auto newOp = builder.create<aig::DelayOp>(
  //           extractOp.getLoc(), extractOp.getType(), newOperands,
  //           delayOp.getDelays());
  //     } else if (auto concatOp =
  //                    extractOp.getInput().getDefiningOp<comb::ConcatOp>())
  //                    {
  //       auto newOperands = concatOp.getOperands();
  //       newOperands[extractOp.getLowBit() / concatOp.getOperands()[0]
  //                                               .getType()
  //                                               .getIntOrFloatBitWidth()] =
  //           extractOp.getInput();
  //     }
  //   }
  // });
  return success();
}

struct LongestPathAnalysisImpl {
  LongestPathAnalysisImpl(mlir::ModuleOp mod,
                          igraph::InstanceGraph *instanceGraph,
                          StringAttr topModuleName)
      : mod(mod), instanceGraph(instanceGraph), topModuleName(topModuleName) {}
  LogicalResult rewrite(mlir::ModuleOp mod);
  LogicalResult rewriteLocal(hw::HWModuleOp mod,
                             mlir::FrozenRewritePatternSet &frozen);
  LogicalResult inlineOnLevel(hw::HWModuleOp mod,
                              mlir::FrozenRewritePatternSet &frozen);

  LogicalResult run();

  mlir::MLIRContext *getContext() { return mod.getContext(); }

private:
  mlir::ModuleOp mod;
  StringAttr topModuleName;
  igraph::InstanceGraph *instanceGraph;
};

struct AndInverterToDelayConversion : OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    // Replace with DelayOp.
    // 1 -> 1, 2 -> 1, 3 -> 2, 4, -> 2
    uint64_t delay = std::max(1u, llvm::Log2_64_Ceil(op.getOperands().size()));
    SmallVector<int64_t> delays(op.getOperands().size(), delay);
    auto delaysAttr = rewriter.getDenseI64ArrayAttr(delays);
    rewriter.replaceOpWithNewOp<aig::DelayOp>(op, op.getType(),
                                              op.getOperands(), delaysAttr);
    return success();
  }
};

struct DelayConcatConversion : OpRewritePattern<aig::DelayOp> {
  using OpRewritePattern<aig::DelayOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DelayOp op,
                                PatternRewriter &rewriter) const override {
    size_t concatIdx = 0;
    comb::ConcatOp concatOp;
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      if ((concatOp = operand.getDefiningOp<comb::ConcatOp>())) {
        concatIdx = i;
        break;
      }
    }

    if (!concatOp)
      return failure();

    SmallVector<Value> newOperands(op.getOperands().take_front(concatIdx));
    newOperands.reserve(op.getNumOperands());
    SmallVector<Value> results;
    results.reserve(concatOp.getNumOperands());
    size_t bitPos = 0;
    for (auto operand : llvm::reverse(concatOp.getOperands())) {
      auto bitWidth = operand.getType().getIntOrFloatBitWidth();
      newOperands.push_back(operand);
      // FIXME: This has to a bug!
      for (auto rest : op.getOperands().drop_front(concatIdx + 1)) {
        // FIXME: Reuse extracted op if possible.
        newOperands.push_back(createOrReuseExtract(rewriter, op.getLoc(), rest,
                                                   bitPos, bitWidth));
      }
      newOperands.push_back(operand);
      bitPos += bitWidth;
      auto result = rewriter.createOrFold<aig::DelayOp>(
          op.getLoc(), operand.getType(), newOperands, op.getDelays());
      results.push_back(result);
      newOperands.resize(concatIdx);
    }

    std::reverse(results.begin(), results.end());
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, op.getType(), results);
    return success();
  }
};

struct ExtractDelayConversion : OpRewritePattern<comb::ExtractOp> {
  using OpRewritePattern<comb::ExtractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: It's not pretty to replicate extract op.
    auto delayOp = op.getInput().getDefiningOp<DelayOp>();
    if (!delayOp)
      return failure();
    SmallVector<Value> newOperands;

    for (auto operand : delayOp.getOperands()) {
      if (operand.getType() != delayOp.getType()) {
        if (operand.getType().isInteger(1)) {
          newOperands.push_back(operand);
          continue;
        } else {
          operand = rewriter.create<comb::ReplicateOp>(
              op.getLoc(), delayOp.getType(), operand);
        }
      }

      newOperands.push_back(
          createOrReuseExtract(rewriter, op.getLoc(), operand, op.getLowBit(),
                               op.getType().getIntOrFloatBitWidth()));
    }
    rewriter.replaceOpWithNewOp<aig::DelayOp>(op, op.getType(), newOperands,
                                              delayOp.getDelays());
    return success();
  }
};

struct ExtractConcatConversion : OpRewritePattern<comb::ExtractOp> {
  using OpRewritePattern<comb::ExtractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType() == op.getInput().getType()) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    // TODO: It's not pretty to replicate extract op.
    auto concatOp = op.getInput().getDefiningOp<comb::ConcatOp>();
    if (!concatOp)
      return failure();
    size_t lowBit = op.getLowBit();
    size_t width = op.getType().getIntOrFloatBitWidth();
    auto newOp =
        createOrReuseExtract(rewriter, op.getLoc(), concatOp, lowBit, width);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct ExtractReplicateConversion : OpRewritePattern<comb::ExtractOp> {
  using OpRewritePattern<comb::ExtractOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::ExtractOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: It's not pretty to replicate extract op.
    auto replicateOp = op.getInput().getDefiningOp<comb::ReplicateOp>();
    if (!replicateOp)
      return failure();
    SmallVector<Value> newOperands;
    size_t lowBit = op.getLowBit();
    size_t width = op.getType().getIntOrFloatBitWidth();
    for (size_t i = 0, e = replicateOp.getMultiple(); i != e; ++i) {
      auto operand = replicateOp.getInput();
      auto opWidth = operand.getType().getIntOrFloatBitWidth();
      if (width == 0)
        break;

      if (lowBit >= opWidth) {
        lowBit -= opWidth;
        continue;
      }

      // lowBit < width
      size_t extractWidth = std::min(opWidth - lowBit, width);
      newOperands.push_back(createOrReuseExtract(rewriter, op.getLoc(), operand,
                                                 lowBit, extractWidth));
      width -= extractWidth;
    }
    std::reverse(newOperands.begin(), newOperands.end());

    auto newOp =
        rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), newOperands);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

LogicalResult
LongestPathAnalysisImpl::rewriteLocal(hw::HWModuleOp mod,
                                      mlir::FrozenRewritePatternSet &frozen) {
  OpBuilder rewriter(getContext());
  rewriter.setInsertionPointToStart(mod.getBodyBlock());
  // for (auto &blockArgument : mod.getBodyBlock()->getArguments()) {
  //   if (failed(replaceWithConcat(blockArgument)))
  //     return failure();
  // }

  // mod.walk([&](Operation *op) {
  //   if (isa<seq::CompRegOp, seq::FirRegOp>(op))
  //     if (failed(replaceWithConcat(op->getResult(0))))
  //       return WalkResult::interrupt();
  //   return WalkResult::advance();
  // });

  RewritePatternSet patterns2(getContext());
  patterns2.add<AndInverterToDelayConversion>(getContext());

  mlir::FrozenRewritePatternSet frozen2(std::move(patterns2));

  // AndInverterToDelayConversion aigToDelay(getContext());
  // mlir::GreedyRewriteConfig config;
  // config.useTopDownTraversal = true;
  // if (failed(mlir::applyPatternsAndFoldGreedily(mod, frozen2, config))) {
  //   llvm::errs() << "Failed to apply patterns and fold greedily\n";
  //   return success();
  // }

  // mlir::PatternRewriter rewriter2(mod.getContext());
  // mlir::DominanceInfo domInfo(mod);
  // mlir::eliminateCommonSubExpressions(rewriter2, domInfo, mod);

  // if (failed(mlir::applyPatternsAndFoldGreedily(mod, frozen, config))) {
  //   llvm::errs() << "Failed to apply patterns and fold greedily\n";

  //   return success();
  // }

  LocalPathAnalysisTransform transform;
  if (failed(transform.run(mod)))
    return failure();
  return success();
}

LogicalResult LongestPathAnalysisImpl::run() {
  RewritePatternSet patterns(getContext());

  patterns
      .add<ShrinkDelayPattern, ExtractDelayConversion, ExtractConcatConversion,
           ExtractReplicateConversion, DelayConcatConversion>(getContext());
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  std::mutex mutex;
  SmallVector<hw::HWModuleOp> underHierarchy;
  for (auto *node : llvm::post_order(instanceGraph->lookup(topModuleName)))
    if (node && node->getModule())
      if (auto hwMod = dyn_cast<hw::HWModuleOp>(*node->getModule()))
        underHierarchy.push_back(hwMod);
  llvm::errs() << "Under hierarchy: " << underHierarchy.size() << "\n";

  auto result = mlir::failableParallelForEach(
      getContext(), underHierarchy, [&](hw::HWModuleOp mod) {
        auto startTime = std::chrono::high_resolution_clock::now();
        {
          std::lock_guard<std::mutex> lock(mutex);
          llvm::errs() << mod.getName() << " start\n";
        }

        auto result = rewriteLocal(mod, frozen);

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

LogicalResult
PrintLongestPathAnalysisPass::rewrite(mlir::ModuleOp mod,
                                      igraph::InstanceGraph &instanceGraph) {
  return success();
}

LogicalResult PrintLongestPathAnalysisPass::rewrite(hw::HWModuleOp mod) {
  return success();
}

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
