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
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Mutex.h"

#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/IR/Builders.h>

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
    DenseMap<Value, SmallVector<int64_t>> retimingLag;
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
    Value clock = nullptr;
    bool clockFound = false;

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
      if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
        if (clockFound) {
          if (clock && firreg.getClk() != clock) {
            firreg->emitError() << "Different clock " << firreg.getClk() << " "
                                << clock << "\n";
            clock = nullptr;
          }
        } else {
          clock = firreg.getClk();
          clockFound = true;
        }
      }
    });

    int64_t maxCost = 0;
    for (auto [bound, cost] : boundaryCost) {
      maxCost = std::max(maxCost, cost);
    }

    // FIXME:
    boundaryCost.clear();
    maxCost = 0;

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
      auto &retimingLag = hwMod.retimingLag;
      retimingLag.clear();
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
                  assert(1 + resultLag - inputLag == 0);

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
                size_t baseCost =
                    std::max(1u, llvm::Log2_64_Ceil(aig->getNumOperands()));
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
                    if (resultLag - inputLag < 0) {
                      llvm::errs()
                          << "Negative lag " << resultLag << " " << inputLag
                          << " " << aig << " " << bitPos << "\n";
                      llvm::errs() << operand << "\n";
                    }
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

                incrementRetimingLag(result, i);

                if (!clock || boundry.count(result))
                  return WalkResult::interrupt();

                // Cannot retime async reset now. It's necessary to be extremely
                // careful when retiming async reset.
                if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
                  if (firreg.getIsAsync())
                    return WalkResult::interrupt();
                }
                exceeded = true;
                changed++;
              }
              if (isa<aig::AndInverterOp>(op)) {
                for (auto operand : op->getOperands()) {
                  auto inputLag = getRetimingLag(operand, i);
                  if (inputLag - getRetimingLag(result, i) > 0) {
                    op->emitError() 
                        << "Negative lag detected: result=" << result << "[" << i << "]"
                        << " (lag=" << getRetimingLag(result, i) << "), "
                        << "input=" << operand << "[" << i << "]"
                        << " (lag=" << inputLag << ")\n"
                        << "Result distance: " << localLongestPath.getMaxDelay(result, i)
                        << ", Input distance: " << localLongestPath.getMaxDelay(operand, i);
                    return WalkResult::interrupt();
                  }
                }
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
  auto retimingTarget = hwMods[maxIndex].result;
  llvm::errs() << "Max index " << maxIndex << " "
               << hwMods[maxIndex].module.getModuleNameAttr()
               << hwMods[maxIndex].result << "\n";
  mlir::failableParallelForEach(ctx, hwMods, [&](auto &hwMod) {
    auto module = hwMod.module;
    auto &cost = hwMod.cost;
    auto &lam = hwMod.am;
    notifyStart(hwMod);
    SetVector<Value> boundry;
    Value clock = nullptr;
    bool clockFound = false;

    module.walk([&](Operation *op) {
      if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
        if (clockFound) {
          if (clock && firreg.getClk() != clock) {
            firreg->emitError() << "Different clock " << firreg.getClk() << " "
                                << clock << "\n";
            clock = nullptr;
          }
        } else {
          clock = firreg.getClk();
          clockFound = true;
        }
      }
    });
    if (!clock)
      return success();

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

    auto &retimingLag = hwMod.retimingLag;
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

    DenseMap<std::tuple<Value, int, int>, Value> laggedValue;
    std::function<Value(Value, int, int)> getStage =
        [&](Value value, int bitPos, int stage) -> Value {
      assert(bitPos == 0);
      if (stage == 0)
        return value;

      auto it = laggedValue.find({value, bitPos, stage});
      if (it != laggedValue.end())
        return it->second;
      auto result = getStage(value, bitPos, stage - 1);
      OpBuilder b(value.getContext());
      b.setInsertionPointAfterValue(result);
      auto reg = b.create<seq::FirRegOp>(value.getLoc(), result, clock,
                                         b.getStringAttr("retimed"));
      laggedValue[{value, bitPos, stage}] = reg;
      retimingLag[reg].resize(getBitWidth(reg), getRetimingLag(result, bitPos));
      return reg;
    };
    llvm::MapVector<Value, Value> mapping;
    SmallVector<Operation *> toProcess;
    module.walk([&](Operation *op) {
      if (isa<seq::FirRegOp, aig::AndInverterOp>(op))
        toProcess.push_back(op);
    });

    for (auto op : toProcess) {
      if (auto firreg = dyn_cast<seq::FirRegOp>(op)) {
        SmallVector<Value> newNexts;
        bool allOne = true, allZero = true;
        for (size_t i = 0, e = getBitWidth(firreg); i < e; ++i) {
          auto lag = getRetimingLag(firreg, i);
          auto next = getRetimingLag(firreg.getNext(), i);
          if (1 + lag - next != 1) {
            allOne = false;
          }
          if (1 + lag - next != 0) {
            allZero = false;
          }
        }
        if (allZero) {
          mapping[firreg] = firreg.getNext();
        }
        if (allOne || allZero)
          continue;
        for (size_t i = 0, e = getBitWidth(firreg); i < e; ++i) {
          auto lag = getRetimingLag(firreg, i);
          auto next = getRetimingLag(firreg.getNext(), i);
          int stage = 1 + lag - next;
          assert(stage > 0);
          auto newNext = getStage(firreg.getNext(), i, stage - 1);
          newNexts.push_back(newNext);
        }
        std::reverse(newNexts.begin(), newNexts.end());
        OpBuilder b(firreg);
        auto concat = b.create<comb::ConcatOp>(firreg->getLoc(), newNexts);
        // TODO: Insert mux if necessary.
        mapping[firreg] = concat;
      }
      if (auto aig = dyn_cast<aig::AndInverterOp>(op)) {
        SmallVector<Value> newOperands;
        for (size_t i = 0, end = op->getNumOperands(); i < end; ++i) {
          auto operand = op->getOperand(i);
          SmallVector<Value> newOperandBits;
          bool needsChange = false;
          for (size_t j = 0, e = getBitWidth(operand); j < e; ++j) {
            auto lag = getRetimingLag(operand, j);
            auto resultLag = getRetimingLag(aig, j);
            int stage = resultLag - lag;
            if (stage < 0) {
              llvm::errs() << "Negative stage " << stage << " for " << aig
                           << " " << j << "\n";
              llvm::errs() << "resultLag " << resultLag << " lag " << lag << " "
                           << operand << "\n";
            }
            assert(stage >= 0);
            if (stage == 0) {
              newOperandBits.push_back(operand);
              continue;
            }
            auto newOperand = getStage(operand, j, stage);
            newOperandBits.push_back(newOperand);
            needsChange = true;
          }
          if (needsChange) {
            auto b = OpBuilder::atBlockBegin(aig->getBlock());
            std::reverse(newOperandBits.begin(), newOperandBits.end());
            auto concat =
                b.createOrFold<comb::ConcatOp>(aig->getLoc(), newOperandBits);
            op->setOperand(i, concat);
          }
        }
      }
    }

    for (auto [old, new_] : mapping.takeVector()) {
      old.replaceAllUsesWith(new_);
      if (auto firreg = dyn_cast<seq::FirRegOp>(old.getDefiningOp()))
        firreg->erase();
    }
    notifyEnd(hwMod);

    return success();
  });
  // mlir::failableParallelForEach(ctx, hwMods,
  //                               [&](auto &hwMod) { hwMod.module->dump();
  //                               });
}
