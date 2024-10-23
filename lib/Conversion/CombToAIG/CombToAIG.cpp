//===- CombToAIG.cpp - Comb to AIG Conversion Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToAIG.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOAIG
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

SmallVector<Value> extractBits(ConversionPatternRewriter &rewriter, Value val) {
  assert(val.getType().isInteger() && "expected integer");
  auto width = val.getType().getIntOrFloatBitWidth();
  SmallVector<Value> bits;
  bits.reserve(width);
  for (int64_t i = 0; i < width; ++i)
    bits.push_back(rewriter.create<comb::ExtractOp>(val.getLoc(), val, i, 1));
  return bits;
}

struct CombMuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // c ? a : b = (replicate(c) & a) | (~replicate(c) & b)
    Value cond = op.getCond();
    auto trueVal = op.getTrueValue();
    auto falseVal = op.getFalseValue();
    if (!op.getType().isInteger(1))
      cond =
          rewriter.create<comb::ReplicateOp>(op.getLoc(), op.getType(), cond);

    auto lhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, trueVal);
    auto rhs = rewriter.create<aig::AndInverterOp>(op.getLoc(), cond, falseVal,
                                                   true, false);
    rewriter.replaceOpWithNewOp<comb::OrOp>(op, lhs, rhs);
    return success();
  }
};

struct CombAddOpConversion : OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputs = adaptor.getInputs();
    if (inputs.size() != 2)
      return failure();

    Value sum, carry;
    auto width = op.getType().getIntOrFloatBitWidth();
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(), 0);
      return success();
    }

    auto lhsBits = extractBits(rewriter, inputs[0]);
    auto rhsBits = extractBits(rewriter, inputs[1]);
    SmallVector<Value> results;
    for (int64_t i = 0; i < width; ++i) {
      SmallVector<Value> range = {lhsBits[i], rhsBits[i]};
      if (carry)
        range.push_back(carry);

      // sum[i] = xor(sum[i-1], a[i], b[i])
      Value xorOp = rewriter.create<comb::XorOp>(op.getLoc(), range, true);
      results.push_back(xorOp);
      if (i == width - 1)
        break;

      // carry[i] = (sum[i-1] & (a[i] ^ b[i])) | (a[i] & b[i])
      Value nextCarry = rewriter.create<comb::AndOp>(
          op.getLoc(), ValueRange{lhsBits[i], rhsBits[i]}, true);
      if (sum) {
        auto aXnorB = rewriter.create<comb::XorOp>(
            op.getLoc(), ValueRange{lhsBits[i], rhsBits[i]}, true);
        auto andOp = rewriter.create<comb::AndOp>(
            op.getLoc(), ValueRange{sum, aXnorB}, true);
        auto orOp = rewriter.create<comb::OrOp>(
            op.getLoc(), ValueRange{andOp, nextCarry}, true);
        nextCarry = orOp;
      }
      sum = xorOp;
      carry = nextCarry;
    }
    // Reverse the results to match the bit order
    std::reverse(results.begin(), results.end());
    rewriter.replaceOpWithNewOp<comb::ConcatOp>(op, results);
    return success();
  }
};

struct CombICmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    switch (op.getPredicate()) {
    default:
      return failure();

    case ICmpPredicate::eq: {
      // a == b  ==> ~(a[n] ^ b[n]) & ~(a[n-1] ^ b[n-1]) & ...
      auto xorOp = rewriter.createOrFold<comb::XorOp>(op.getLoc(), lhs, rhs);
      auto xorBits = extractBits(rewriter, xorOp);
      SmallVector<bool> allInverts(xorBits.size(), true);
      rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, xorBits, allInverts);
      return success();
    }

    case ICmpPredicate::ne: {
      // a != b  ==> (a[n] ^ b[n]) | (a[n-1] ^ b[n-1]) | ...
      auto xorOp = rewriter.createOrFold<comb::XorOp>(op.getLoc(), lhs, rhs);
      rewriter.replaceOpWithNewOp<comb::OrOp>(op, extractBits(rewriter, xorOp),
                                              true);
      return success();
    }

    case ICmpPredicate::uge:
    case ICmpPredicate::ugt:
    case ICmpPredicate::ule:
    case ICmpPredicate::ult: {
      bool isLess = op.getPredicate() == ICmpPredicate::ult ||
                    op.getPredicate() == ICmpPredicate::ule;
      bool includeEq = op.getPredicate() == ICmpPredicate::uge ||
                       op.getPredicate() == ICmpPredicate::ule;
      // a <= b  ==> ( a[n] & ~b[n]) | (a[n] & b[n] & a[n-1:0] <= b[n-1:0])
      // a <  b  ==> ( a[n] & ~b[n]) | (a[n] & b[n] & a[n-1:0] < b[n-1:0])
      // a >= b  ==> (~a[n] &  b[n]) | (a[n] & b[n] & a[n-1:0] >= b[n-1:0])
      // a >  b  ==> (~a[n] &  b[n]) | (a[n] & b[n] & a[n-1:0] > b[n-1:0])

      auto width = op.getType().getIntOrFloatBitWidth();
      Value acc =
          rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), includeEq);

      for (int64_t i = 0; i < width; ++i) {
        auto aBit =
            rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, i, 1);
        auto bBit =
            rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), rhs, i, 1);
        auto aBitNotBBit = rewriter.createOrFold<aig::AndInverterOp>(
            op.getLoc(), aBit, bBit, !isLess, isLess);

        auto aBitAndBBit = rewriter.createOrFold<comb::AndOp>(
            op.getLoc(), ValueRange{aBit, bBit, acc}, true);
        acc = rewriter.createOrFold<comb::OrOp>(op.getLoc(), aBitNotBBit,
                                                aBitAndBBit, true);
      }
      rewriter.replaceOp(op, acc);
      return success();
    }
    }
  }
};

struct CombShlOpConversion : OpConversionPattern<ShlOp> {
  using OpConversionPattern<ShlOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto bits = extractBits(rewriter, rhs);
    auto width = op.getType().getIntOrFloatBitWidth();
    auto leafSize = llvm::PowerOf2Ceil(width);
    auto allZero = rewriter.create<hw::ConstantOp>(
        op.getLoc(), IntegerType::get(rewriter.getContext(), width), 0);

    // This represents a binary tree. The leaf nodes are the bits of the input
    // value. The internal nodes are the muxes.
    SmallVector<Value> nodes;
    nodes.resize(2 * leafSize - 1, allZero);

    // Initialize nodes[leafSize - 1 : 2*leafSize - 1] which represents the leaf
    // nodes.
    for (int64_t i = 0; i < width; ++i) {
      auto zeros = rewriter.createOrFold<hw::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), i), 0);
      auto extract =
          rewriter.createOrFold<comb::ExtractOp>(op.getLoc(), lhs, 0, width - i);
      auto concat =
          rewriter.createOrFold<comb::ConcatOp>(op.getLoc(), extract, zeros);
      nodes[leafSize + i - 1] = concat;
    }

    for (int64_t i = leafSize - 1; i > 0; --i) {
      // 3, 2, 1
      // 0, 0, 1
      auto selector =
          bits[llvm::Log2_64_Ceil(width) - llvm::Log2_64_Ceil(i + 1)];
      auto mux = rewriter.createOrFold<comb::MuxOp>(op.getLoc(), selector,
                                              nodes[2 * i], nodes[2 * i - 1]);
      nodes[i - 1] = mux;
    }
    rewriter.replaceOp(op, nodes[0]);
    return success();
  }
};

/// Lower a comb::AndOp operation to aig::AndInverterOp
struct CombAndOpConversion : OpConversionPattern<AndOp> {
  using OpConversionPattern<AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<bool> nonInverts(adaptor.getInputs().size(), false);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, adaptor.getInputs(),
                                                    nonInverts);
    return success();
  }
};

/// Lower a comb::OrOp operation to aig::AndInverterOp with invert flags
struct CombOrOpConversion : OpConversionPattern<OrOp> {
  using OpConversionPattern<OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Implement Or using And and invert flags: a | b = ~(~a & ~b)
    SmallVector<bool> allInverts(adaptor.getInputs().size(), true);
    auto andOp = rewriter.create<aig::AndInverterOp>(
        op.getLoc(), adaptor.getInputs(), allInverts);
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, andOp,
                                                    /*invert=*/true);
    return success();
  }
};

/// Lower a comb::XorOp operation to AIG operations
struct CombXorOpConversion : OpConversionPattern<XorOp> {
  using OpConversionPattern<XorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(XorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Xor using And with invert flags: a ^ b = (a | b) & (~a | ~b)
    // (a | b) = ~(~a & ~b)
    // (~a | ~b) = ~(a & b)
    auto inputs = adaptor.getInputs();
    SmallVector<bool> allInverts(inputs.size(), true);
    SmallVector<bool> allNotInverts(inputs.size(), false);

    // a | b = ~(~a & ~b)
    auto notAAndNotB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allInverts);
    auto aAndB =
        rewriter.create<aig::AndInverterOp>(op.getLoc(), inputs, allNotInverts);

    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(op, notAAndNotB, aAndB,
                                                    /*lhs_invert=*/true,
                                                    /*rhs_invert=*/true);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertCombToAIGPass
    : public impl::ConvertCombToAIGBase<ConvertCombToAIGPass> {

  void runOnOperation() override;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::ConvertCombToAIGBase;
  using ConvertCombToAIGBase<ConvertCombToAIGPass>::keepBitwiseLogic;
};
} // namespace

static void populateCombToAIGConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<CombAndOpConversion, CombOrOpConversion, CombXorOpConversion,
               CombMuxOpConversion, CombAddOpConversion, CombICmpOpConversion,
               CombShlOpConversion>(patterns.getContext());
}

void ConvertCombToAIGPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<comb::CombDialect>();
  // Extract, Concat, and Replicate are legal in the output
  target.addLegalOp<comb::ExtractOp, comb::ConcatOp, comb::ReplicateOp,
                    hw::ConstantOp>();
  target.addLegalDialect<aig::AIGDialect>();

  // This is test only option to keep bitwise logic instead of lowering to AIG.
  if (keepBitwiseLogic) {
    target.addLegalOp<comb::MuxOp, comb::XorOp, comb::AndOp, comb::OrOp>();
  }

  RewritePatternSet patterns(&getContext());
  populateCombToAIGConversionPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
