//===- LowerWordToBits.cpp - Bit-Blasting Words to Bits ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers multi-bit AIG operations to single-bit ones.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/Builders.h>

#define DEBUG_TYPE "aig-lower-word-to-bits"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_LOWERWORDTOBITS
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Lower Word to Bits pass
//===----------------------------------------------------------------------===//

static size_t getBitWidth(Value value) {
  assert(value);
  if (auto vecType = dyn_cast<seq::ClockType>(value.getType()))
    return 1;
  if (auto memory = dyn_cast<seq::FirMemType>(value.getType()))
    return memory.getWidth();
  return hw::getBitWidth(value.getType());
}

namespace {
struct LowerWordToBitsPass
    : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
  void runOnOperation() override;

  llvm::MapVector<Value, SmallVector<Value>> processedOps;
  ArrayRef<Value> lower(Value value);
  Value getBit(Value value, size_t index);
};
} // namespace

Value LowerWordToBitsPass::getBit(Value value, size_t index) {
  // llvm::errs() << "value= " << value << " index= " << index << "\n";
  if (getBitWidth(value) <= 1)
    return value;

  auto *op = value.getDefiningOp();
  if (getBitWidth(value) <= index) {
    llvm::dbgs() << "getBitWidth(value) = " << getBitWidth(value)
                 << " index = " << index << "\n";
    llvm::dbgs() << "value = " << value << "\n";
    assert(false && "index is out of range");
  }
  if (!op)
    return lower(value)[index];

  return TypeSwitch<Operation *, Value>(op)
      .Case<comb::ConcatOp>([&](comb::ConcatOp op) {
        for (auto operand : llvm::reverse(op.getOperands())) {
          auto width = getBitWidth(operand);
          if (index < width)
            return getBit(operand, index);
          index -= width;
        }
        assert(false && "index is out of range");
      })
      .Case<comb::ExtractOp>([&](comb::ExtractOp ext) {
        return getBit(ext.getInput(), ext.getLowBit() + index);
      })
      .Case<comb::ReplicateOp>([&](comb::ReplicateOp op) {
        return getBit(op.getInput(), index % getBitWidth(op.getOperand()));
      })
      .Default([&](auto op) { return lower(value)[index]; });
}

ArrayRef<Value> LowerWordToBitsPass::lower(Value value) {
  auto *it = processedOps.find(value);
  if (it != processedOps.end()) {
    if (getBitWidth(value) != it->second.size()) {
      llvm::dbgs() << "getBitWidth(value) = " << getBitWidth(value)
                   << " it->second.size() = " << it->second.size() << "\n";
      llvm::dbgs() << "value = " << value << "\n";
    }
    assert(getBitWidth(value) == it->second.size());
    return it->second;
  }
  auto width = getBitWidth(value);
  if (width <= 1) {
    processedOps.insert({value, {value}});
    return processedOps[value];
  }

  OpBuilder builder(&getContext());
  builder.setInsertionPointAfterValue(value);
  auto *op = value.getDefiningOp();
  SmallVector<Value> results;
  results.reserve(width);
  if (!op) {
    comb::extractBits(builder, value, results);
    assert(results.size() == width);
    auto resultIt = processedOps.insert({value, results});
    assert(resultIt.second);
    return resultIt.first->second;
  }

  TypeSwitch<Operation *, void>(op)
      .Case<comb::AndOp, comb::OrOp, comb::XorOp>([&](auto op) {
        for (int64_t i = 0; i < width; i++) {
          SmallVector<Value> operands;
          operands.reserve(op->getNumOperands());
          for (auto operand : op->getOperands()) {
            auto bits = getBit(operand, i);
            operands.push_back(bits);
          }
          results.push_back(
              builder.create<decltype(op)>(op->getLoc(), operands, true));
        }
        processedOps.insert({value, results});
      })
      .Case<aig::AndInverterOp>([&](aig::AndInverterOp op) {
        for (int64_t i = 0; i < width; i++) {
          SmallVector<Value> operands;
          operands.reserve(op->getNumOperands());
          for (auto operand : op->getOperands()) {
            auto bits = getBit(operand, i);
            operands.push_back(bits);
          }
          results.push_back(builder.create<aig::AndInverterOp>(
              op->getLoc(), operands, op.getInvertedAttr()));
        }
        assert(results.size() == width);

        auto resultIt = processedOps.insert({value, results});
        assert(resultIt.second);
      })
      .Case<seq::FirRegOp>([&](seq::FirRegOp op) {
        if (op.getPreset()) {
          // If there is a preset, we don't lower the register.
          // TODO: Fix it.
          comb::extractBits(builder, op, results);
          processedOps.insert({value, results});
          return;
        }

        SmallVector<Value> resetValues;
        if (op.getResetValue())
          comb::extractBits(builder, op.getResetValue(), resetValues);

        // Extract the next value.
        // NOTE: Don't use lower at this point.
        SmallVector<Value> nextValues;
        comb::extractBits(builder, op.getNext(), nextValues);

        assert(nextValues.size() == width);
        for (int64_t i = 0; i < width; i++) {
          auto name = op.getNameAttr();
          if (name)
            name = builder.getStringAttr(name.getValue() + "[" +
                                         std::to_string(i) + "]");
          seq::FirRegOp seqReg;
          if (op.getReset()) {
            seqReg = builder.create<seq::FirRegOp>(
                op->getLoc(), nextValues[i], op.getClk(), name, op.getReset(),
                resetValues[i]);
          } else {
            seqReg = builder.create<seq::FirRegOp>(op->getLoc(), nextValues[i],
                                                   op.getClk(), name);
          }

          seqReg.setIsAsync(op.getIsAsync());
          results.push_back(seqReg);
        }
        assert(results.size() == width);

        processedOps.insert({value, results});
      })
      .Default([&](auto op) {
        SmallVector<Value> results;
        comb::extractBits(builder, value, results);
        assert(results.size() == width);
        processedOps.insert({value, results});
      });

  auto resultIt = processedOps.find(value);
  assert(resultIt != processedOps.end());
  assert(resultIt->second.size() == getBitWidth(value));
  return resultIt->second;
}

void LowerWordToBitsPass::runOnOperation() {
  auto operaton = getOperation();
  operaton.walk([&](Operation *op) {
    if (isa<aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp,
            seq::FirRegOp>(op)) {
      auto it = lower(op->getResult(0));
      if (it.size() != getBitWidth(op->getResult(0))) {
        llvm::dbgs() << "it.size() = " << it.size()
                     << " getBitWidth(op->getResult(0)) = "
                     << getBitWidth(op->getResult(0)) << "\n";
        llvm::dbgs() << "op = " << op << "\n";
      }
    }
  });


  for (auto &[value, results] : llvm::make_early_inc_range(processedOps)) {
    if (getBitWidth(value) <= 1)
      continue;

    auto *op = value.getDefiningOp();
    if (!op)
      continue;

    if (value.use_empty()) {
      op->erase();
      continue;
    }
    if (isa<aig::AndInverterOp, comb::AndOp, comb::OrOp, comb::XorOp,
            seq::FirRegOp>(op)) {
      mlir::OpBuilder builder(op);
      std::reverse(results.begin(), results.end());
      auto concat = builder.create<comb::ConcatOp>(value.getLoc(), results);
      value.replaceAllUsesWith(concat);
      op->erase();
    }
  }
  
  // Make sure clear the data structure.
  processedOps.clear();
}