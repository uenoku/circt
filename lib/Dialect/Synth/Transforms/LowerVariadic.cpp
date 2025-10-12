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

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/PriorityQueue.h"
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>

#define DEBUG_TYPE "synth-lower-variadic"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LOWERVARIADIC
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

namespace {
Value lowerVariadicAndInverterOp(aig::AndInverterOp op, OperandRange operands,
                                 ArrayRef<bool> inverts,
                                 mlir::IRRewriter &rewriter) {
  switch (operands.size()) {
  case 0:
    assert(0 && "cannot be called with empty operand range");
    break;
  case 1:
    if (inverts[0])
      return aig::AndInverterOp::create(rewriter, op.getLoc(), operands[0],
                                        true);
    else
      return operands[0];
  case 2:
    return aig::AndInverterOp::create(rewriter, op.getLoc(), operands[0],
                                      operands[1], inverts[0], inverts[1]);
  default:
    auto firstHalf = operands.size() / 2;
    auto lhs =
        lowerVariadicAndInverterOp(op, operands.take_front(firstHalf),
                                   inverts.take_front(firstHalf), rewriter);
    auto rhs =
        lowerVariadicAndInverterOp(op, operands.drop_front(firstHalf),
                                   inverts.drop_front(firstHalf), rewriter);
    return aig::AndInverterOp::create(rewriter, op.getLoc(), lhs, rhs);
  }

  return Value();
}

} // namespace

//===----------------------------------------------------------------------===//
// Lower Variadic pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerVariadicPass : public impl::LowerVariadicBase<LowerVariadicPass> {
  using LowerVariadicBase::LowerVariadicBase;
  void runOnOperation() override;
};
} // namespace

struct ValueWithArrivalTime {
  Value value;
  DelayType arrivalTime;
  ValueWithArrivalTime(Value value, DelayType arrivalTime)
      : value(value), arrivalTime(arrivalTime) {}

  bool operator<(const ValueWithArrivalTime &other) const {
    return arrivalTime < other.arrivalTime;
  }
  bool operator>(const ValueWithArrivalTime &other) const {
    return arrivalTime > other.arrivalTime;
  }
};

void LowerVariadicPass::runOnOperation() {
  auto topo = synth::topologicallySortGraphRegionBlocks(
      getOperation(), [](Value value, mlir::Operation *op) -> bool {
        return !isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
            op->getDialect());
      });

  if (failed(topo))
    return signalPassFailure();

  synth::IncrementalLongestPathAnalysis *analysis =
      &getAnalysis<synth::IncrementalLongestPathAnalysis>();
  auto op = getOperation();
  SmallVector<mlir::OperationName> names;
  for (auto name : opNames)
    names.push_back(mlir::OperationName(name, &getContext()));
  auto shouldLower = [&](Operation *op) {
    if (names.empty())
      return true;
    return llvm::find(names, op->getName()) != names.end();
  };
  mlir::IRRewriter rewriter(&getContext());
  rewriter.setListener(analysis);
  auto result = op->walk([&](Operation *op) {
    if (shouldLower(op)) {
      rewriter.setInsertionPoint(op);
      if (op->getNumOperands() <= 2)
        return WalkResult::advance();
      if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
        auto newOp = lowerVariadicAndInverterOp(
            andInverterOp, andInverterOp->getOperands(),
            andInverterOp.getInverted(), rewriter);
        rewriter.replaceOp(op, newOp);
        return WalkResult::advance();
      }
      // and, or, xor, mul, add etc.
      if (op->hasTrait<mlir::OpTrait::IsCommutative>()) {
        // Check if the operation has more than 2 operands.
        llvm::PriorityQueue<ValueWithArrivalTime,
                            std::vector<ValueWithArrivalTime>,
                            std::greater<ValueWithArrivalTime>>
            queue;
        auto enqueue = [&](Value value) -> LogicalResult {
          auto delay = analysis->getMaxDelay(value);
          if (failed(delay))
            return failure();
          queue.push(ValueWithArrivalTime(value, *delay));
          return success();
        };
        for (auto operand : op->getOperands())
          if (failed(enqueue(operand)))
            return WalkResult::interrupt();

        // Lower the operation.
        rewriter.setInsertionPoint(op);
        while (queue.size() >= 2) {
          auto lhs = queue.top();
          queue.pop();
          auto rhs = queue.top();
          queue.pop();
          OperationState state(op->getLoc(), op->getName());
          state.addOperands(ValueRange{lhs.value, rhs.value});
          state.addTypes(op->getResult(0).getType());
          auto *newOp = Operation::create(state);
          rewriter.insert(newOp);
          if (failed(enqueue(newOp->getResult(0))))
            return WalkResult::interrupt();
        }
        rewriter.replaceOp(op, queue.top().value);
        return WalkResult::advance();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
