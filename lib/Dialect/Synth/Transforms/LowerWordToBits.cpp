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

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synth-lower-word-to-bits"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LOWERWORDTOBITS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;
//
// //===----------------------------------------------------------------------===//
// // Rewrite patterns
// //===----------------------------------------------------------------------===//
//
// namespace {
// template <typename OpTy>
// struct WordRewritePattern : public OpRewritePattern<OpTy> {
//   using OpRewritePattern<OpTy>::OpRewritePattern;
//
//   LogicalResult matchAndRewrite(OpTy op,
//                                 PatternRewriter &rewriter) const override {
//     auto width = op.getType().getIntOrFloatBitWidth();
//     if (width <= 1)
//       return failure();
//
//     SmallVector<Value> results;
//     // We iterate over the width in reverse order to match the endianness of
//     // `comb.concat`.
//     for (int64_t i = width - 1; i >= 0; --i) {
//       SmallVector<Value> operands;
//       for (auto operand : op.getOperands()) {
//         // Reuse bits if we can extract from `comb.concat` operands.
//         if (auto concat = operand.template getDefiningOp<comb::ConcatOp>()) {
//           // For the simplicity, we only handle the case where all the
//           // `comb.concat` operands are single-bit.
//           if (concat.getNumOperands() == width &&
//               llvm::all_of(concat.getOperandTypes(), [](Type type) {
//                 return type.getIntOrFloatBitWidth() == 1;
//               })) {
//             // Be careful with the endianness here.
//             operands.push_back(concat.getOperand(width - i - 1));
//             continue;
//           }
//         }
//         // Otherwise, we need to extract the bit.
//         operands.push_back(
//             comb::ExtractOp::create(rewriter, op.getLoc(), operand, i, 1));
//       }
//       results.push_back(
//           OpTy::create(rewriter, op.getLoc(), operands,
//           op.getInvertedAttr()));
//     }
//
//     rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, results);
//     return success();
//   }
// };
//
// } // namespace
//
// //===----------------------------------------------------------------------===//
// // Lower Word to Bits pass
// //===----------------------------------------------------------------------===//
//
// namespace {
// struct LowerWordToBitsPass
//     : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
//   void runOnOperation() override;
// };
// } // namespace
//
// void LowerWordToBitsPass::runOnOperation() {
//   RewritePatternSet patterns(&getContext());
//   patterns.add<WordRewritePattern<aig::AndInverterOp>,
//                WordRewritePattern<mig::MajorityInverterOp>>(&getContext());
//
//   mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
//   mlir::GreedyRewriteConfig config;
//   // Use top-down traversal to reuse bits from `comb.concat`.
//   config.setUseTopDownTraversal(true);
//
//   if (failed(
//           mlir::applyPatternsGreedily(getOperation(), frozenPatterns,
//           config)))
//     return signalPassFailure();
// }

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

struct Driver {
  Driver(Block *block) : block(block) {}
  llvm::MapVector<Value, SmallVector<Value>> processedOps;
  llvm::MapVector<Value, llvm::KnownBits> knownBits;
  ArrayRef<Value> lower(Value value);
  Value getBit(Value value, size_t index);

  const llvm::KnownBits &getKnownBits(Value value);

  Value getConstant(bool value);
  Block *block;
  Value constant[2];
  size_t numLoweredBits = 0;
  size_t numLoweredConstants = 0;
  size_t numLoweredOps = 0;
};
struct LowerWordToBitsPass
    : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
  void runOnOperation() override;
};
} // namespace

/*
struct UnknownBitIterator {
  uint64_t pos = 0;
  uint64_t width = 0;
  const uint64_t *data;

  UnknownBitIterator(uint64_t width, APInt &unknownBits)
      : width(width), data(unknownBits.getRawData()) {}
  // Iterator interface
  UnknownBitIterator &operator++() {
    pos++;
    while (true) {
      uint64_t word = data[pos / 64];
      uint64_t rest = pos % 64;
      if (word & (1ULL << rest))
        break;
      pos++;
      if (pos >= width)
        break;
    }
    return *this;
  }
  bool operator!=(const UnknownBitIterator &other) const {
    return pos != other.pos;
  }
  uint64_t operator*() const { return data[pos / 64] & (1ULL << (pos % 64)); }
};
  */

const llvm::KnownBits &Driver::getKnownBits(Value value) {
  auto *it = knownBits.find(value);
  if (it != knownBits.end())
    return it->second;
  auto width = getBitWidth(value);
  auto *op = value.getDefiningOp();
  if (!op) {
    return knownBits.insert({value, llvm::KnownBits(width)}).first->second;
  }

  if (auto aig = dyn_cast<aig::AndInverterOp>(op)) {
    llvm::KnownBits known(width);
    // Initialize to all ones for AND operation
    known.One = APInt::getAllOnes(width);
    known.Zero = APInt::getZero(width);

    for (auto [operand, inverted] :
         llvm::zip(aig.getInputs(), aig.getInverted())) {
      auto operandKnown = getKnownBits(operand);
      if (inverted)
        // Complement the known bits by swapping Zero and One
        std::swap(operandKnown.Zero, operandKnown.One);
      known &= operandKnown;
    }
    return knownBits.insert({value, known}).first->second;
  }

  if (auto mig = dyn_cast<mig::MajorityInverterOp>(op)) {
    // Give up if it's not a 3-input majority inverter.
    if (mig.getNumOperands() != 3)
      return knownBits.insert({value, llvm::KnownBits(width)}).first->second;

    std::array<llvm::KnownBits, 3> operandKnown;
    for (auto [operandBits, operand, inverted] :
         llvm::zip(operandKnown, mig.getInputs(), mig.getInverted())) {
      operandBits = getKnownBits(operand);
      // Complement the known bits by swapping Zero and One
      if (inverted)
        std::swap(operandBits.Zero, operandBits.One);
    }

    auto known = (operandKnown[0] & operandKnown[1]) |
                 (operandKnown[0] & operandKnown[2]) |
                 (operandKnown[1] & operandKnown[2]);
    return knownBits.insert({value, known}).first->second;
  }

  return knownBits.insert({value, comb::computeKnownBits(value)}).first->second;
}

Value Driver::getBit(Value value, size_t index) {
  // llvm::errs() << "value= " << value << " index= " << index << "\n";
  if (getBitWidth(value) <= 1)
    return value;

  auto known = getKnownBits(value);
  if (known.Zero[index] || known.One[index])
    return getConstant(known.One[index]);

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

ArrayRef<Value> Driver::lower(Value value) {
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

  OpBuilder builder(value.getContext());
  builder.setInsertionPointAfterValue(value);
  auto *op = value.getDefiningOp();
  SmallVector<Value> results;
  if (!op) {
    comb::extractBits(builder, value, results);
    assert(results.size() == width);
    auto resultIt = processedOps.insert({value, results});
    assert(resultIt.second);
    return resultIt.first->second;
  }

  auto known = getKnownBits(value);
  APInt knownMask = known.Zero | known.One;
  llvm::BitVector unknownBits(knownMask);
  numLoweredConstants += knownMask.popcount();
  numLoweredBits += width;
  numLoweredOps++;

  TypeSwitch<Operation *, void>(op)
      .Case<comb::AndOp, comb::OrOp, comb::XorOp>([&](auto op) {
        for (size_t i = 0; i < width; i++) {
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
      .Case<aig::AndInverterOp, mig::MajorityInverterOp>([&](auto op) {
        // Iterate know unknown positions.
        size_t pos = 0;
        SmallVector<Value> results(width, nullptr);
        for (size_t i = 0; i < width; i++) {
          if (knownMask[i]) {
            results[i] = getConstant(known.One[i]);
            continue;
          }
          SmallVector<Value> operands;
          operands.reserve(op->getNumOperands());
          for (auto operand : op->getOperands()) {
            auto bits = getBit(operand, pos);
            operands.push_back(bits);
          }
          results[i] = builder.createOrFold<decltype(op)>(
              op->getLoc(), operands, op.getInvertedAttr());
          if (auto name =
                  op->template getAttrOfType<StringAttr>("sv.namehint")) {
            auto newName =
                StringAttr::get(op.getContext(), name.getValue() + "[" +
                                                     std::to_string(pos) + "]");
            if (auto orig = results[i].getDefiningOp())
              orig->setAttr("sv.namehint", newName);
          }
          pos++;
        }

        assert(results.size() == width);
        auto resultIt = processedOps.insert({value, std::move(results)});
        assert(resultIt.second);
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
  Driver driver(getOperation().getBodyBlock());
  auto operation = getOperation();
  operation.walk([&](Operation *op) {
    if (isa<aig::AndInverterOp, mig::MajorityInverterOp, comb::AndOp,
            comb::OrOp, comb::XorOp>(op)) {
      auto it = driver.lower(op->getResult(0));
      if (it.size() != getBitWidth(op->getResult(0))) {
        llvm::dbgs() << "it.size() = " << it.size()
                     << " getBitWidth(op->getResult(0)) = "
                     << getBitWidth(op->getResult(0)) << "\n";
        llvm::dbgs() << "op = " << op << "\n";
      }
    }
  });

  for (auto &[value, results] :
       llvm::make_early_inc_range(driver.processedOps)) {
    if (getBitWidth(value) <= 1)
      continue;

    auto *op = value.getDefiningOp();
    if (!op)
      continue;

    if (value.use_empty()) {
      op->erase();
      continue;
    }
    if (isa<aig::AndInverterOp, mig::MajorityInverterOp, comb::AndOp,
            comb::OrOp, comb::XorOp>(op)) {
      mlir::OpBuilder builder(op);
      std::reverse(results.begin(), results.end());
      auto concat = builder.create<comb::ConcatOp>(value.getLoc(), results);
      value.replaceAllUsesWith(concat);
      op->erase();
    }
  }

  numLoweredBits += driver.numLoweredBits;
  numLoweredConstants += driver.numLoweredConstants;
  numLoweredOps += driver.numLoweredOps;

  // Make sure clear the data structure.
  driver.processedOps.clear();
}
Value Driver::getConstant(bool value) {
  if (!constant[value]) {
    auto builder = OpBuilder::atBlockBegin(block);
    constant[value] = builder.create<hw::ConstantOp>(
        builder.getUnknownLoc(), builder.getI1Type(), value);
  }
  return constant[value];
}
