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

static size_t getBitWidth(Value value) {
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
  };
  SmallVector<LocalState> hwMods;
  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    if (!longestPath.isAnalysisAvailable(hwMod.getModuleNameAttr()))
      continue;
    hwMods.emplace_back(hwMod, am);
  }

  // 1. Compute cost for all values in the module.
  mlir::failableParallelForEach(ctx, hwMods, [&](auto &hwMod) {
    auto module = hwMod.module;
    auto &cost = hwMod.cost;
    auto &lam = hwMod.am;
    DenseMap<std::pair<Value, int>, int64_t> boundaryCost;
    for (auto arg : module.getBodyBlock()->getArguments())
      for (size_t i = 0, e = getBitWidth(arg); i < e; ++i)
        boundaryCost[{arg, i}] = longestPath.getMaxDelay(arg, i);

    module.walk([&](Operation *op) {
      if (auto instance = dyn_cast<hw::InstanceOp>(op))
        for (auto result : instance.getResults())
          for (size_t i = 0, e = getBitWidth(result); i < e; ++i)
            boundaryCost[{result, i}] = longestPath.getMaxDelay(result, i);
    });

    auto tryThreshold = [&](int64_t threshold) {
      DenseMap<std::pair<Value, int>, int64_t>
          retimingLag; // function `r` in the paper

      int64_t iterationLimit = 100;
      while (iterationLimit-- > 0) {
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
                auto resultLag = retimingLag[{value, bitPos}];
                auto inputLag = retimingLag[{next, bitPos}];
                // w(u ->' v) := w(u -> v) + r(v) - r(u) = 1 + resultLag -
                // inputLag
                if (1 + resultLag - inputLag > 0) {
                  // We still have a register here. So fallback to the
                  // original.
                  return false;
                } else {
                  assert(1 + resultLag - inputLag == 0);

                  // Ok, let's ignore register. In that case we can visit the
                  // next.
                  auto result = visit(next, bitPos, results);
                  if (failed(result))
                    return failure();
                  return true;
                }
              }

              // If aig node.
              if (auto aig =
                      dyn_cast<aig::AndInverterOp>(value.getDefiningOp())) {

                auto resultLag = retimingLag[{value, bitPos}];
                size_t baseCost = llvm::Log2_64_Ceil(aig->getNumOperands());
                bool existRegisterOnEdge =
                    llvm::any_of(aig->getOperands(), [&](Value operand) {
                      auto inputLag = retimingLag[{operand, bitPos}];
                      return resultLag - inputLag > 0;
                    });
                // Use default.
                if (!existRegisterOnEdge)
                  return false;
                for (auto operand : aig->getOperands()) {
                  // w(u -> v) := w(u -> v) + r(v) - r(u) = 0 + resultLag -
                  // inputLag
                  auto inputLag = retimingLag[{operand, bitPos}];
                  if (resultLag - inputLag > 0) {
                    // Ok, we have a register here.
                    results.push_back(OpenPath({}, value, bitPos, baseCost));
                  } else {
                    assert(resultLag - inputLag == 0);
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

        bool exceeded = false;
        size_t changed = 0;
        auto result = module.walk([&](Operation *op) {
          for (auto result : op->getResults())
            for (size_t i = 0, e = getBitWidth(result); i < e; ++i) {
              auto delay = localLongestPath.getMaxDelay(result, i);

              if (delay > threshold) {
                // llvm::errs() << "[Retiming] " << iterationLimit << " "
                //              << threshold << ": Delay " << delay << " for "
                //              << result << " " << i << "\n";

                exceeded = true;
                retimingLag[{result, i}]++;
                changed++;
              }
            }

          return WalkResult::advance();
        });
        if (!exceeded) {
          return true;
        }
      }
      return false;
    };

    int64_t upperLimit = 1000, lowerLimit = 0;
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

    llvm::errs() << "Final result " << upperLimit << "\n";

    return success();
  });
}
