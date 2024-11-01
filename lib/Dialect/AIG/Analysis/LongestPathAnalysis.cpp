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
    }
    if (newOperands.size() == 1)
      return newOperands.front();

    std::reverse(newOperands.begin(), newOperands.end());
    return rewriter.create<comb::ConcatOp>(loc, newOperands);
  }

  return rewriter.createOrFold<comb::ExtractOp>(
      loc, rewriter.getIntegerType(width), operand, lowBit);
}

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
  DenseMap<BitRange, Value> bitRangeToValue;
  Value getOrCreateBitRange(OpBuilder &builder, Value operand, size_t lowBit,
                            size_t width);
  LogicalResult lower(OpBuilder &builder, aig::AndInverterOp op);

  LogicalResult run(hw::HWModuleOp mod);
};

LogicalResult LocalPathAnalysisTransform::lower(OpBuilder &builder,
                                                aig::AndInverterOp op) {
  auto loc = op.getLoc();
  builder.setInsertionPoint(op);
  // Replace with DelayOp.
  // 1 -> 1, 2 -> 1, 3 -> 2, 4, -> 2
  uint64_t delay = std::max(1u, llvm::Log2_64_Ceil(op.getOperands().size()));
  SmallVector<int64_t> delays(op.getOperands().size(), delay);
  auto delaysAttr = builder.getDenseI64ArrayAttr(delays);
  Value delayOp = builder.create<aig::DelayOp>(loc, op.getType(),
                                               op.getOperands(), delaysAttr);

  op.getResult().replaceAllUsesWith(delayOp);
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

static bool isRootValue(Value value) {
  if (auto arg = value.dyn_cast<BlockArgument>())
    return true;
  return isa<seq::CompRegOp, seq::FirRegOp>(value.getDefiningOp());
}
/*
LogicalResult LocalPathAnalysisTransform::run(hw::HWModuleOp mod) {
  // NOTE: The result IR could be huge so instead of relying on
  // GreedyRewriteDriver and DialectConversion manually iterate over the
  // operations.

  SmallVector<Operation *> needToLegalize;
  OpBuilder builder(mod.getContext());
  // 1. Replace AndInverterOp with DelayOp.
  mod.walk([&](AndInverterOp op) {
    if (failed(lower(builder, op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  DenseSet<Value> legalizedValues;

  mod.walk([&](Operation *op) {
    // 2. Hoist Concat and Sink ExtractOp.
    if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
      if (isRootValue(extractOp.getInput())) {
        legalizedValues.insert(extractOp.getInput());
        return WalkResult::advance();
      }

      // ExtractOp(AndDelay(a, b, c)) -> AndDelay(Extract(b))
      if (auto delayOp = extractOp.getInput().getDefiningOp<aig::DelayOp>()) {
        auto newOperands = delayOp.getOperands();
        newOperands[extractOp.getLowBit() / delayOp.getDelays()[0]] =
            extractOp.getInput();
        auto newOp = builder.create<aig::DelayOp>(
            extractOp.getLoc(), extractOp.getType(), newOperands,
            delayOp.getDelays());
      } else if (auto concatOp =
                     extractOp.getInput().getDefiningOp<comb::ConcatOp>()) {
        auto newOperands = concatOp.getOperands();
        newOperands[extractOp.getLowBit() / concatOp.getOperands()[0]
                                                .getType()
                                                .getIntOrFloatBitWidth()] =
            extractOp.getInput();
      }
    }
  });
}*/

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

struct ShrinkDelayPattern : OpRewritePattern<aig::DelayOp> {
  using OpRewritePattern<aig::DelayOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DelayOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<int64_t> newDelays;
    SmallVector<Value> newOperands;
    DenseMap<std::tuple<Value, size_t, size_t>, size_t> operandToIdx;
    SmallVector<std::pair<Value, int64_t>> worklist;
    DenseMap<Value, int64_t> observedMaxDelay;

    bool changed = false;
    for (auto [operand, delay] :
         llvm::reverse(llvm::zip(op.getOperands(), op.getDelays())))
      worklist.push_back({operand, delay});

    while (!worklist.empty()) {
      auto [operand, delay] = worklist.pop_back_val();

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
          worklist.push_back({dOpOperand, dOpDelay + delay});
          changed = true;
        }
      } else if (auto constantOp = operand.getDefiningOp<hw::ConstantOp>()) {
        // Delay of constant is 0.
        changed = true;
      } else if (auto replicateOp =
                     operand.getDefiningOp<comb::ReplicateOp>()) {
        worklist.push_back({replicateOp.getInput(), delay});
      } else {
        std::tuple<Value, size_t, size_t> operandTuple;
        if (auto extractOp = operand.getDefiningOp<comb::ExtractOp>()) {
          operandTuple =
              std::make_tuple(operand, extractOp.getLowBit(),
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
          newOperands.push_back(createOrReuseExtract(
              rewriter, op.getLoc(), operand, std::get<1>(operandTuple),
              std::get<2>(operandTuple)));
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
      for (auto rest : op.getOperands().drop_front(concatIdx + 1)) {
        // FIXME: Reuse extracted op if possible.
        newOperands.push_back(createOrReuseExtract(rewriter, op.getLoc(), rest,
                                                   bitPos, bitWidth));
      }
      newOperands.push_back(operand);
      bitPos += bitWidth;
      auto result = rewriter.createOrFold<aig::DelayOp>(
          op.getLoc(), op.getType(), newOperands, op.getDelays());
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
      if (operand.getType() != op.getType()) {
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
  auto replaceWithConcat = [&](Value value) {
    SmallVector<Value> newOperands;

    if (!value.getType().isInteger()) {
      return success(isa<seq::ClockType>(value.getType()));
    }

    for (size_t i = 0, e = value.getType().getIntOrFloatBitWidth(); i != e;
         ++i) {
      newOperands.push_back(
          createOrReuseExtract(rewriter, value.getLoc(), value, i, 1));
    }
    std::reverse(newOperands.begin(), newOperands.end());
    auto newOp = rewriter.create<comb::ConcatOp>(value.getLoc(), newOperands);
    value.replaceUsesWithIf(newOp, [&](OpOperand &use) {
      return !isa<comb::ExtractOp>(use.getOwner());
    });
    return success();
  };

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
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  if (failed(mlir::applyPatternsAndFoldGreedily(mod, frozen2, config)))
    return failure();

  mlir::PatternRewriter rewriter2(mod.getContext());
  mlir::DominanceInfo domInfo(mod);
  mlir::eliminateCommonSubExpressions(rewriter2, domInfo, mod);

  if (failed(mlir::applyPatternsAndFoldGreedily(mod, frozen, config)))
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
  if (succeeded(result))
    return failure();
  return success();
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
  if (failed(longestPathAnalysis.run()))
    return signalPassFailure();

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
