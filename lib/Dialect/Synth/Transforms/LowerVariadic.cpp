//===- LowerVariadic.cpp - Lowering Variadic to Binary Ops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers variadic operations to binary operations using a
// delay-aware algorithm for commutative operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PriorityQueue.h"

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

struct ValueWithArrivalTime {
  llvm::PointerIntPair<Value, 1, bool> value;
  int64_t arrivalTime;
  ValueWithArrivalTime(Value value, int64_t arrivalTime)
      : value(value), arrivalTime(arrivalTime) {}
  ValueWithArrivalTime(Value value, int64_t arrivalTime, bool invert)
      : value(value, invert), arrivalTime(arrivalTime) {}
  Value getValue() const { return value.getPointer(); }
  bool isInverted() const { return value.getInt(); }

  bool operator>(const ValueWithArrivalTime &other) const {
    return arrivalTime > other.arrivalTime;
  }
};

struct LowerVariadicPass : public impl::LowerVariadicBase<LowerVariadicPass> {
  using LowerVariadicBase::LowerVariadicBase;
  void runOnOperation() override;
};

} // namespace

FailureOr<Value> constructBalancedTree(
    Operation *op, llvm::function_ref<bool(OpOperand &)> isInverted,
    llvm::function_ref<FailureOr<ValueWithArrivalTime>(Value value,
                                                       bool invert)>
        enqueue,
    llvm::function_ref<Value(ValueWithArrivalTime, ValueWithArrivalTime)>
        create) {
  // Priority queue.
  llvm::PriorityQueue<ValueWithArrivalTime, std::vector<ValueWithArrivalTime>,
                      std::greater<ValueWithArrivalTime>>
      queue;
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    auto inverted = isInverted(op->getOpOperand(i));
    auto result = enqueue(op->getOperand(i), inverted);
    if (failed(result))
      return failure();
    queue.push(*result);
  }

  while (queue.size() >= 2) {
    auto lhs = queue.top();
    queue.pop();
    auto rhs = queue.top();
    queue.pop();
    auto result = enqueue(create(lhs, rhs), /*inverted=*/false);
    if (failed(result))
      return failure();
    queue.push(*result);
  }

  return queue.top().getValue();
}

void LowerVariadicPass::runOnOperation() {
  // Topologically sort operations in graph regions to ensure operands are
  // defined before uses.
  if (failed(synth::topologicallySortGraphRegionBlocks(
          getOperation(), [](Value, Operation *op) -> bool {
            return !isa_and_nonnull<comb::CombDialect, synth::SynthDialect>(
                op->getDialect());
          })))
    return signalPassFailure();

  auto *analysis = &getAnalysis<synth::IncrementalLongestPathAnalysis>();
  auto moduleOp = getOperation();

  // Build set of operation names to lower if specified.
  SmallVector<OperationName> names;
  for (const auto &name : opNames)
    names.push_back(OperationName(name, &getContext()));

  // Return true if the operation should be lowered.
  auto shouldLower = [&](Operation *op) {
    if (names.empty())
      return true;
    return llvm::find(names, op->getName()) != names.end();
  };

  mlir::IRRewriter rewriter(&getContext());
  rewriter.setListener(analysis);

  auto result = moduleOp->walk([&](Operation *op) {
    if (!shouldLower(op) || op->getNumOperands() <= 2)
      return WalkResult::advance();

    rewriter.setInsertionPoint(op);

    // Handle AndInverterOp specially due to inversion flags.
    if (auto andInverterOp = dyn_cast<aig::AndInverterOp>(op)) {
      auto result = constructBalancedTree(
          op,
          [&](OpOperand &operand) {
            return andInverterOp.isInverted(operand.getOperandNumber());
          },
          [&](Value value, bool invert) -> FailureOr<ValueWithArrivalTime> {
            auto delay = analysis->getMaxDelay(value);
            if (failed(delay))
              return failure();

            llvm::errs() << value << " " << *delay << "\n";
            return ValueWithArrivalTime(value, *delay, invert);
          },
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            return rewriter.create<aig::AndInverterOp>(
                op->getLoc(), lhs.getValue(), rhs.getValue(), lhs.isInverted(),
                rhs.isInverted());
          });
      if (failed(result))
        return WalkResult::interrupt();
      rewriter.replaceOp(op, *result);
      return WalkResult::advance();
    }

    // Handle commutative operations (and, or, xor, mul, add, etc.) using
    // delay-aware lowering to minimize critical path.
    if (op->hasTrait<OpTrait::IsCommutative>()) {
      auto result = constructBalancedTree(
          op, [&](OpOperand &operand) { return false; },
          [&](Value value, bool invert) -> FailureOr<ValueWithArrivalTime> {
            auto delay = analysis->getMaxDelay(value);
            if (failed(delay))
              return failure();
            return ValueWithArrivalTime(value, *delay);
          },
          [&](ValueWithArrivalTime lhs, ValueWithArrivalTime rhs) {
            OperationState state(op->getLoc(), op->getName());
            state.addOperands(ValueRange{lhs.getValue(), rhs.getValue()});
            state.addTypes(op->getResult(0).getType());
            auto *newOp = Operation::create(state);
            rewriter.insert(newOp);
            return newOp->getResult(0);
          });
      if (failed(result))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}
