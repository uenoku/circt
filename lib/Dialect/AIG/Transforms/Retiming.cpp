//===- LowerVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic AndInverter operations to binary AndInverter
// operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGAnalysis.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Mutex.h"

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <mlir/Analysis/TopologicalSortUtils.h>

#define DEBUG_TYPE "aig-lower-variadic"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_RETIMING
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lower Variadic pass
//===----------------------------------------------------------------------===//

namespace {
struct RetimingPass : public impl::RetimingBase<RetimingPass> {
  void runOnOperation() override;
};
} // namespace

static int64_t getBitWidth(Value value) {
  if (auto vecType = dyn_cast<seq::ClockType>(value.getType()))
    return 1;
  if (auto memory = dyn_cast<seq::FirMemType>(value.getType()))
    return memory.getWidth();
  return hw::getBitWidth(value.getType());
}

void RetimingPass::runOnOperation() {
  auto &longestPath = getAnalysis<circt::aig::LongestPathAnalysis>();
  auto am = getAnalysisManager();

  auto module = getOperation();
  auto *ctx = &getContext();
  struct LocalState {
    hw::HWModuleOp module;
    mlir::AnalysisManager am;
    LocalState(hw::HWModuleOp module, mlir::AnalysisManager &mam)
        : module(module), am(mam.nest(module)) {}
    DenseMap<std::pair<Value, int64_t>, int64_t> cost;
    int64_t result = 0;
  };
  SmallVector<LocalState> hwMods;
  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    if (!longestPath.isAnalysisAvailable(hwMod.getModuleNameAttr()))
      continue;
    hwMods.emplace_back(hwMod, am);
  }
  llvm::sys::SmartMutex<true> mutex;
  SetVector<StringAttr> running;
  auto notifyStart = [&](auto &hwMod) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.insert(hwMod.module.getModuleNameAttr());
    llvm::errs() << "[Retiming] " << hwMod.module.getModuleNameAttr()
                 << " started. running=[";
    for (auto &name : running)
      llvm::errs() << name << " ";
    llvm::errs() << "]\n";
  };

  auto notifyEnd = [&](auto &hwMod) {
    std::lock_guard<llvm::sys::SmartMutex<true>> lock(mutex);
    running.remove(hwMod.module.getModuleNameAttr());
    llvm::errs() << "[Retiming] " << hwMod.module.getModuleNameAttr()
                 << " finished. running=[";
    for (auto &name : running)
      llvm::errs() << name << " ";
    llvm::errs() << "]\n";
  };

  // 1. Compute cost for all values in the module.
  mlir::failableParallelForEach(ctx, hwMods, [&](auto &hwMod) {
    auto module = hwMod.module;
    auto &cost = hwMod.cost;
    auto &lam = hwMod.am;
    notifyStart(hwMod);
    DenseMap<std::pair<Value, int>, int64_t> boundaryCost;
    for (auto arg : module.getBodyBlock()->getArguments())
      for (size_t i = 0, e = getBitWidth(arg); i < e; ++i)
        boundaryCost[{arg, i}] = longestPath.getMaxDelay(arg, i);
    SetVector<Value> boundry;

    module.walk([&](Operation *op) {
      if (auto instance = dyn_cast<hw::InstanceOp>(op)) {
        for (auto result : instance.getResults())
          for (size_t i = 0, e = getBitWidth(result); i < e; ++i)
            boundaryCost[{result, i}] = longestPath.getMaxDelay(result, i);

        for (auto operand : instance->getOperands())
          if (longestPath.isAnalysisAvailable(
                  instance.getReferencedModuleNameAttr())) {
            boundry.insert(operand);
          }
      }
      if (auto output = dyn_cast<hw::OutputOp>(op)) {
        for (auto operand : output->getOperands())
          boundry.insert(operand);
      }
    });

    int64_t maxCost = 0;
    for (auto [bound, cost] : boundaryCost) {
      maxCost = std::max(maxCost, cost);
    }

    auto terminator = module.getBodyBlock()->getTerminator();
    OpBuilder builder(terminator);
    // llvm::SetVector<Value> wires;
    // for (auto result : terminator->getOperands()) {
    //   // Add a wire.
    //   auto wire = builder.create<hw::WireOp>(result.getLoc(), result);
    //   wires.insert(wire);
    // }
    // Placeholder for output wire.
    // terminator->setOperands(wires.getArrayRef());

    auto tryThreshold = [&](int64_t threshold, bool commit = false) {
      if (threshold < maxCost)
        return false;
      DenseMap<Value, SmallVector<int64_t>>
          retimingLag; // function `r` in the paper
      auto getRetimingLag = [&](Value value, size_t bitPos) -> int64_t {
        if (!retimingLag.count(value))
          return 0;
        return retimingLag[value][bitPos];
      };

      auto incrementRetimingLag = [&](Value value, size_t bitPos) {
        if (!retimingLag.count(value))
          retimingLag[value].resize(getBitWidth(value), 0);
        retimingLag[value][bitPos]++;
      };

      int64_t iterationLimit =
          longestPath.getNumNodes(module.getModuleNameAttr());
      size_t counter = 0;
      while (iterationLimit-- > 0) {
        counter++;
        // llvm::errs() << "[Retiming] " << hwMod.module.getModuleNameAttr()
        //              << "iteration " << iterationLimit << "\n";
        LongestPathAnalysis localLongestPath(
            module, lam,
            [&](llvm::function_ref<LogicalResult(Value, size_t,
                                                 SmallVectorImpl<OpenPath> &)>
                    visit,
                Value value, size_t bitPos,
                SmallVectorImpl<OpenPath> &results) -> FailureOr<bool> {
              // If value is blockarg, use pre-saved cost.
              if (auto arg = dyn_cast<BlockArgument>(value)) {
                results.push_back(
                    OpenPath({}, arg, bitPos, boundaryCost[{arg, bitPos}]));
                return true;
              }
              if (auto op = dyn_cast<hw::InstanceOp>(value.getDefiningOp())) {
                results.push_back(
                    OpenPath({}, value, bitPos, boundaryCost[{value, bitPos}]));
                return true;
              }
              // %0 = aig.and_inv > r(%0) = 0
              // %1 = aig.and_inv %0 > r(%0) = 1
              // %2 = firreg %1 = r(%1) = 0

              // If firreg.
              if (auto reg = dyn_cast<seq::FirRegOp>(value.getDefiningOp())) {
                // Check if the reg is deleted.
                auto next = reg.getNext();
                auto resultLag = getRetimingLag(value, bitPos);
                auto inputLag = getRetimingLag(next, bitPos);
                // w(u ->' v) := w(u -> v) + r(v) - r(u) = 1 + resultLag -
                // inputLag
                if (1 + resultLag - inputLag > 0) {
                  LLVM_DEBUG({
                    llvm::errs()
                        << "[Retiming] " << counter << " " << "Register " << reg
                        << " " << bitPos << " is not deleted\n";
                  });
                  // We still have a register here. So fallback to the
                  // original.
                  return false;
                } else {
                  LLVM_DEBUG({
                    llvm::errs()
                        << "[Retiming] " << counter << " " << "Register " << reg
                        << " " << bitPos << " is deleted\n";
                  });
                  // assert(1 + resultLag - inputLag == 0);

                  // Ok, let's ignore register. In that case we can visit the
                  // next.
                  auto result = visit(next, bitPos, results);

                  if (failed(result))
                    return failure();
                  if (reg.getResetValue()) {
                    if (failed(visit(reg.getResetValue(), bitPos, results)))
                      return failure();
                  }
                  return true;
                }
              }

              // If aig node.
              if (auto aig =
                      dyn_cast<aig::AndInverterOp>(value.getDefiningOp())) {

                auto resultLag = getRetimingLag(value, bitPos);
                size_t baseCost = llvm::Log2_64_Ceil(aig->getNumOperands());
                bool existRegisterOnEdge =
                    llvm::any_of(aig->getOperands(), [&](Value operand) {
                      auto inputLag = getRetimingLag(operand, bitPos);
                      return resultLag - inputLag > 0;
                    });
                // Use default.
                if (!existRegisterOnEdge)
                  return false;
                for (auto operand : aig->getOperands()) {
                  // w(u -> v) := w(u -> v) + r(v) - r(u) = 0 + resultLag -
                  // inputLag
                  auto inputLag = getRetimingLag(operand, bitPos);
                  if (resultLag - inputLag > 0) {
                    // Ok, we have a register here.
                    results.push_back(OpenPath({}, value, bitPos, baseCost));
                  } else {
                    // assert(resultLag - inputLag == 0);
                    size_t oldIndex = results.size();
                    auto result = visit(operand, bitPos, results);
                    if (failed(result))
                      return failure();
                    for (auto i = oldIndex, e = results.size(); i < e; ++i)
                      results[i].delay += baseCost;
                  }
                }
                return true;
              }

              return false;
            });

        llvm::errs() << "Retiming initialized done "
                     << hwMod.module.getModuleNameAttr() << " " << counter
                     << "\n";

        bool exceeded = false;
        size_t changed = 0;
        size_t opCount = 0;
        auto result = module.walk([&](Operation *op) {
          SmallVector<Attribute> attr, attr2;
          for (auto result : op->getResults()) {
            if (getBitWidth(result) < 0)
              continue;
            for (size_t i = 0, e = getBitWidth(result); i < e; ++i) {
              opCount++;
              if (hwMod.module.getModuleNameAttr().getValue() ==
                  "SiFive_ComputeCluster_Tile_MSHR") {
                if (opCount % 1000 == 0) {
                  llvm::errs() << "opCount " << opCount << " "
                               << module.getModuleNameAttr() << "\n";
                }
              }

              auto delay = localLongestPath.getMaxDelay(result, i);
              attr.push_back(
                  IntegerAttr::get(IntegerType::get(ctx, 32), delay));

              attr2.push_back(IntegerAttr::get(IntegerType::get(ctx, 32),
                                               getRetimingLag(result, i)));

              if (delay > threshold) {
                // llvm::errs() << "[Retiming] " << iterationLimit << " "
                //              << threshold << ": Delay " << delay << " for "
                //              << result << " " << i << "\n";

                if (boundry.count(result))
                  return WalkResult::interrupt();
                exceeded = true;
                incrementRetimingLag(result, i);
                changed++;
              }
            }
          }

          // op->setAttr("aig.max.delay", ArrayAttr::get(ctx, attr));
          // op->setAttr("aig.retiming", ArrayAttr::get(ctx, attr2));
          return WalkResult::advance();
        });
        LLVM_DEBUG({
          llvm::errs() << "// ======= Iteration " << counter << " changed "
                       << changed << "\n";
          module->dump();
          llvm::errs() << "// ======= Iteration " << counter << " end \n";
        });
        if (result.wasInterrupted())
          return false;
        if (!exceeded) {
          if (commit) {
            module.walk([&](Operation *op) {
              if (op->getNumResults() == 1) {
                SmallVector<Attribute> attr;
                for (auto result : op->getResults()) {
                  for (int64_t i = 0, e = getBitWidth(result); i < e; ++i) {
                    attr.push_back(IntegerAttr::get(IntegerType::get(ctx, 32),
                                                    getRetimingLag(result, i)));
                  }
                }
                op->setAttr("aig.retiming", ArrayAttr::get(ctx, attr));
              }
            });
          }
          return true;
        }
      }
      return false;
    };

    int64_t upperLimit = 256, lowerLimit = 0;
    while (upperLimit - lowerLimit > 1) {
      int64_t mid = (upperLimit + lowerLimit) / 2;
      llvm::errs() << "Trying threshold " << mid << "[" << lowerLimit << ","
                   << upperLimit << "]\n";
      auto result = tryThreshold(mid);

      llvm::errs() << "Result " << result << " for " << mid << "\n";
      if (result)
        upperLimit = mid;
      else
        lowerLimit = mid;
    }

    hwMod.result = upperLimit;
    tryThreshold(upperLimit, /*commit=*/true);

    llvm::errs() << "Final result " << upperLimit << "\n";
    notifyEnd(hwMod);

    return success();
  });
  int64_t maxIndex = 0;
  for (auto [index, hwMod] : llvm::enumerate(hwMods)) {
    if (hwMod.result > hwMods[maxIndex].result)
      maxIndex = index;
  }
  llvm::errs() << "Max index " << maxIndex << " "
               << hwMods[maxIndex].module.getModuleNameAttr()
               << hwMods[maxIndex].result << "\n";
}
