//===- CombToSMT.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTCOMBTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace comb;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a comb::ReplicateOp operation to smt::RepeatOp
struct CombReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<smt::RepeatOp>(op, op.getMultiple(),
                                               adaptor.getInput());
    return success();
  }
};

/// Lower a comb::ICmpOp operation to a smt::BVCmpOp, smt::EqOp or
/// smt::DistinctOp
struct IcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getPredicate() == ICmpPredicate::weq ||
        adaptor.getPredicate() == ICmpPredicate::ceq ||
        adaptor.getPredicate() == ICmpPredicate::wne ||
        adaptor.getPredicate() == ICmpPredicate::cne)
      return rewriter.notifyMatchFailure(op,
                                         "comparison predicate not supported");

    Value result;
    if (adaptor.getPredicate() == ICmpPredicate::eq) {
      result = smt::EqOp::create(rewriter, op.getLoc(), adaptor.getLhs(),
                                 adaptor.getRhs());
    } else if (adaptor.getPredicate() == ICmpPredicate::ne) {
      result = smt::DistinctOp::create(rewriter, op.getLoc(), adaptor.getLhs(),
                                       adaptor.getRhs());
    } else {
      smt::BVCmpPredicate pred;
      switch (adaptor.getPredicate()) {
      case ICmpPredicate::sge:
        pred = smt::BVCmpPredicate::sge;
        break;
      case ICmpPredicate::sgt:
        pred = smt::BVCmpPredicate::sgt;
        break;
      case ICmpPredicate::sle:
        pred = smt::BVCmpPredicate::sle;
        break;
      case ICmpPredicate::slt:
        pred = smt::BVCmpPredicate::slt;
        break;
      case ICmpPredicate::uge:
        pred = smt::BVCmpPredicate::uge;
        break;
      case ICmpPredicate::ugt:
        pred = smt::BVCmpPredicate::ugt;
        break;
      case ICmpPredicate::ule:
        pred = smt::BVCmpPredicate::ule;
        break;
      case ICmpPredicate::ult:
        pred = smt::BVCmpPredicate::ult;
        break;
      default:
        llvm_unreachable("all cases handled above");
      }

      result = smt::BVCmpOp::create(rewriter, op.getLoc(), pred,
                                    adaptor.getLhs(), adaptor.getRhs());
    }

    Value convVal = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), typeConverter->convertType(op.getType()),
        result);
    if (!convVal)
      return failure();

    rewriter.replaceOp(op, convVal);
    return success();
  }
};

/// Lower a comb::ExtractOp operation to an smt::ExtractOp
struct ExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<smt::ExtractOp>(
        op, typeConverter->convertType(op.getResult().getType()),
        adaptor.getLowBitAttr(), adaptor.getInput());
    return success();
  }
};

/// Lower a comb::MuxOp operation to an smt::IteOp
struct MuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value condition = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getCond());
    rewriter.replaceOpWithNewOp<smt::IteOp>(
        op, condition, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// Lower a comb::SubOp operation to an smt::BVNegOp + smt::BVAddOp
struct SubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value negRhs =
        smt::BVNegOp::create(rewriter, op.getLoc(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<smt::BVAddOp>(op, adaptor.getLhs(), negRhs);
    return success();
  }
};

/// Lower a comb::ParityOp operation to a chain of smt::Extract + XOr ops
struct ParityOpConversion : OpConversionPattern<ParityOp> {
  using OpConversionPattern<ParityOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParityOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned bitwidth =
        cast<smt::BitVectorType>(adaptor.getInput().getType()).getWidth();

    // Note: the SMT bitvector type does not support 0 bitwidth vectors and thus
    // the type conversion should already fail.
    Type oneBitTy = smt::BitVectorType::get(getContext(), 1);
    Value runner =
        smt::ExtractOp::create(rewriter, loc, oneBitTy, 0, adaptor.getInput());
    for (unsigned i = 1; i < bitwidth; ++i) {
      Value ext = smt::ExtractOp::create(rewriter, loc, oneBitTy, i,
                                         adaptor.getInput());
      runner = smt::BVXOrOp::create(rewriter, loc, runner, ext);
    }

    rewriter.replaceOp(op, runner);
    return success();
  }
};

/// Lower a comb::ReverseOp operation to a chain of smt::Extract + Concat ops
struct ReverseOpConversion : OpConversionPattern<ReverseOp> {
  using OpConversionPattern<ReverseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned bitwidth =
        cast<smt::BitVectorType>(adaptor.getInput().getType()).getWidth();

    Type oneBitTy = smt::BitVectorType::get(getContext(), 1);
    // Extract bit 0 (LSB), which becomes the MSB of the reversed result.
    Value result =
        smt::ExtractOp::create(rewriter, loc, oneBitTy, 0, adaptor.getInput());
    // Concatenate remaining bits in ascending order (LSB-to-MSB of input
    // becomes MSB-to-LSB of output).
    for (unsigned i = 1; i < bitwidth; ++i) {
      Value ext = smt::ExtractOp::create(rewriter, loc, oneBitTy, i,
                                         adaptor.getInput());
      result = smt::ConcatOp::create(rewriter, loc, result, ext);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower the SourceOp to the TargetOp one-to-one.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<TargetOp>(
        op,
        OpConversionPattern<SourceOp>::typeConverter->convertType(
            op.getResult().getType()),
        adaptor.getOperands());
    return success();
  }
};

/// Lower the SourceOp to the TargetOp special-casing if the second operand is
/// zero to return a new symbolic value.
template <typename SourceOp, typename TargetOp>
struct DivisionOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = dyn_cast<smt::BitVectorType>(adaptor.getRhs().getType());
    if (!type)
      return failure();

    auto resultType = OpConversionPattern<SourceOp>::typeConverter->convertType(
        op.getResult().getType());
    Value zero =
        smt::BVConstantOp::create(rewriter, loc, APInt(type.getWidth(), 0));
    Value isZero = smt::EqOp::create(rewriter, loc, adaptor.getRhs(), zero);
    Value symbolicVal = smt::DeclareFunOp::create(rewriter, loc, resultType);
    Value division =
        TargetOp::create(rewriter, loc, resultType, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, isZero, symbolicVal, division);
    return success();
  }
};

/// Converts an operation with a variadic number of operands to a chain of
/// binary operations assuming left-associativity of the operation.
template <typename SourceOp, typename TargetOp>
struct VariadicToBinaryOpConversion : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    if (operands.size() < 2)
      return failure();

    Value runner = operands[0];
    for (Value operand : operands.drop_front())
      runner = TargetOp::create(rewriter, op.getLoc(), runner, operand);

    rewriter.replaceOp(op, runner);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Four-state (0,1,X,Z) encoding helpers and conversions
// Encoding: N-bit comb value -> 2*N-bit BV (high N = poison, low N = value).
// (poison,value) = 00->0, 01->1, 10->X, 11->Z. All 2*N-bit patterns valid.
// X/Z are treated as unknown for most ops (Verilog-like), except case/wildcard
// equality which distinguish/mask them. Arithmetic with any unknown bit
// yields all-X, except shifts which shift poison/value when amount is known.
//===----------------------------------------------------------------------===//

namespace fourState {
using namespace mlir;
using namespace circt;

static unsigned getOrigWidth(Type fourStateTy) {
  auto bv = cast<smt::BitVectorType>(fourStateTy);
  assert(bv.getWidth() % 2 == 0);
  return bv.getWidth() / 2;
}
static unsigned getOrigWidth(Value v) {
  return getOrigWidth(v.getType());
}
static smt::BitVectorType bvTy(MLIRContext *ctx, unsigned w) {
  return smt::BitVectorType::get(ctx, w);
}
static Value bvZero(OpBuilder &b, Location loc, unsigned w) {
  if (w == 0)
    return Value{};
  return smt::BVConstantOp::create(b, loc, APInt(w, 0));
}
static Value bvOnes(OpBuilder &b, Location loc, unsigned w) {
  if (w == 0)
    return Value{};
  return smt::BVConstantOp::create(b, loc, APInt::getAllOnes(w));
}
static std::pair<Value, Value> splitPair(OpBuilder &b, Location loc, Value v,
                                         unsigned origW) {
  auto *ctx = b.getContext();
  Value val = smt::ExtractOp::create(b, loc, bvTy(ctx, origW), 0, v);
  Value poison =
      smt::ExtractOp::create(b, loc, bvTy(ctx, origW), origW, v);
  return {val, poison};
}
static Value makePair(OpBuilder &b, Location loc, Value val, Value poison) {
  return smt::ConcatOp::create(b, loc, poison, val);
}
static Value makeKnown(OpBuilder &b, Location loc, Value val, unsigned w) {
  return makePair(b, loc, val, bvZero(b, loc, w));
}
static Value makeAllX(OpBuilder &b, Location loc, unsigned origW) {
  return makePair(b, loc, bvZero(b, loc, origW), bvOnes(b, loc, origW));
}
static Value hasUnknown(OpBuilder &b, Location loc, Value poison) {
  unsigned w = cast<smt::BitVectorType>(poison.getType()).getWidth();
  Value zero = bvZero(b, loc, w);
  return smt::DistinctOp::create(b, loc, poison, zero);
}
static Value bvAnd(OpBuilder &b, Location loc, Value a, Value b2) {
  return smt::BVAndOp::create(b, loc, a, b2);
}
static Value bvOr(OpBuilder &b, Location loc, Value a, Value b2) {
  return smt::BVOrOp::create(b, loc, a, b2);
}
static Value bvXor(OpBuilder &b, Location loc, Value a, Value b2) {
  return smt::BVXOrOp::create(b, loc, a, b2);
}
static Value bvNot(OpBuilder &b, Location loc, Value a) {
  return smt::BVNotOp::create(b, loc, a);
}

/// 4-state mux: cond is BV2, t/f are BV(2*N). Implements Verilog ?: with X:
/// known select picks side, unknown select yields per-bit agreement else X.
static Value buildMux(OpBuilder &b, Location loc, Value condPair, Value tPair,
                      Value fPair, unsigned origW) {
  auto *ctx = b.getContext();
  auto [cVal, cPoison] = splitPair(b, loc, condPair, 1);
  auto [tVal, tPoison] = splitPair(b, loc, tPair, origW);
  auto [fVal, fPoison] = splitPair(b, loc, fPair, origW);
  Value zero1 = bvZero(b, loc, 1);
  Value one1 = smt::BVConstantOp::create(b, loc, 1, 1);
  Value cIsZero = smt::EqOp::create(b, loc, cPoison, zero1);
  Value cValIsOne = smt::EqOp::create(b, loc, cVal, one1);
  Value cValIsZero = smt::EqOp::create(b, loc, cVal, zero1);
  Value isOne = smt::AndOp::create(b, loc, cIsZero, cValIsOne);
  Value isZero = smt::AndOp::create(b, loc, cIsZero, cValIsZero);
  Value isX = smt::DistinctOp::create(b, loc, cPoison, zero1);
  Value zeroN = bvZero(b, loc, origW);
  Value onesN = bvOnes(b, loc, origW);
  Value maskOne = smt::IteOp::create(b, loc, isOne, onesN, zeroN);
  Value maskZero = smt::IteOp::create(b, loc, isZero, onesN, zeroN);
  Value maskX = smt::IteOp::create(b, loc, isX, onesN, zeroN);
  // Per-bit agreement for X-select.
  Value diffVal = bvXor(b, loc, tVal, fVal);
  Value diffPoison = bvXor(b, loc, tPoison, fPoison);
  Value diff = bvOr(b, loc, diffVal, diffPoison);
  Value agree = bvNot(b, loc, diff);
  Value poisXCase = bvOr(b, loc, tPoison, diff);
  Value valXCase = bvAnd(b, loc, tVal, agree);
  Value resVal = bvOr(b, loc,
                      bvOr(b, loc, bvAnd(b, loc, maskOne, tVal),
                           bvAnd(b, loc, maskZero, fVal)),
                      bvAnd(b, loc, maskX, valXCase));
  Value resPoison = bvOr(b, loc,
                         bvOr(b, loc, bvAnd(b, loc, maskOne, tPoison),
                              bvAnd(b, loc, maskZero, fPoison)),
                         bvAnd(b, loc, maskX, poisXCase));
  return makePair(b, loc, resVal, resPoison);
}

} // namespace fourState

//===----------------------------------------------------------------------===//
// Convert Comb to SMT pass
//===----------------------------------------------------------------------===//

// Four-state patterns (to be inserted)
namespace {
struct FourStateReplicateOpConversion : OpConversionPattern<ReplicateOp> {
  using OpConversionPattern<ReplicateOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReplicateOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned inW = cast<IntegerType>(op.getInput().getType()).getWidth();
    if (inW == 0) return failure();
    auto [val, poison] = fourState::splitPair(rewriter, loc, adaptor.getInput(), inW);
    Value repVal = smt::RepeatOp::create(rewriter, loc, op.getMultiple(), val);
    Value repPoison = smt::RepeatOp::create(rewriter, loc, op.getMultiple(), poison);
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, repVal, repPoison));
    return success();
  }
};
struct FourStateExtractOpConversion : OpConversionPattern<ExtractOp> {
  using OpConversionPattern<ExtractOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned inW = cast<IntegerType>(op.getInput().getType()).getWidth();
    unsigned outW = cast<IntegerType>(op.getType()).getWidth();
    auto [val, poison] = fourState::splitPair(rewriter, loc, adaptor.getInput(), inW);
    auto *ctx = rewriter.getContext();
    Value outVal = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, outW), op.getLowBit(), val);
    Value outPoison = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, outW), op.getLowBit(), poison);
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, outVal, outPoison));
    return success();
  }
};
struct FourStateConcatOpConversion : OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (adaptor.getInputs().empty()) return failure();
    Value accVal, accPoison;
    for (auto [idx, inPair] : llvm::enumerate(adaptor.getInputs())) {
      unsigned w = cast<IntegerType>(op.getInputs()[idx].getType()).getWidth();
      auto [val, poison] = fourState::splitPair(rewriter, loc, inPair, w);
      if (idx == 0) { accVal = val; accPoison = poison; }
      else {
        accVal = smt::ConcatOp::create(rewriter, loc, accVal, val);
        accPoison = smt::ConcatOp::create(rewriter, loc, accPoison, poison);
      }
    }
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, accVal, accPoison));
    return success();
  }
};
struct FourStateAndOpConversion : OpConversionPattern<AndOp> {
  using OpConversionPattern<AndOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    if (w == 0 || adaptor.getInputs().empty()) return failure();
    if (adaptor.getInputs().size() == 1) { rewriter.replaceOp(op, adaptor.getInputs()[0]); return success(); }
    Value hasZero, allOne, andVals;
    for (auto [i, inPair] : llvm::enumerate(adaptor.getInputs())) {
      auto [val, poison] = fourState::splitPair(rewriter, loc, inPair, w);
      Value notPoison = fourState::bvNot(rewriter, loc, poison);
      Value notVal = fourState::bvNot(rewriter, loc, val);
      Value knownZero = fourState::bvAnd(rewriter, loc, notPoison, notVal);
      Value knownOne = fourState::bvAnd(rewriter, loc, notPoison, val);
      if (i == 0) { hasZero = knownZero; allOne = knownOne; andVals = val; }
      else {
        hasZero = fourState::bvOr(rewriter, loc, hasZero, knownZero);
        allOne = fourState::bvAnd(rewriter, loc, allOne, knownOne);
        andVals = fourState::bvAnd(rewriter, loc, andVals, val);
      }
    }
    Value known = fourState::bvOr(rewriter, loc, hasZero, allOne);
    Value resPoison = fourState::bvNot(rewriter, loc, known);
    Value resVal = fourState::bvAnd(rewriter, loc, andVals, fourState::bvNot(rewriter, loc, resPoison));
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, resVal, resPoison));
    return success();
  }
};
struct FourStateOrOpConversion : OpConversionPattern<OrOp> {
  using OpConversionPattern<OrOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(OrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    if (w == 0 || adaptor.getInputs().empty()) return failure();
    if (adaptor.getInputs().size() == 1) { rewriter.replaceOp(op, adaptor.getInputs()[0]); return success(); }
    Value hasOne, allZero, orVals;
    for (auto [i, inPair] : llvm::enumerate(adaptor.getInputs())) {
      auto [val, poison] = fourState::splitPair(rewriter, loc, inPair, w);
      Value notPoison = fourState::bvNot(rewriter, loc, poison);
      Value notVal = fourState::bvNot(rewriter, loc, val);
      Value knownOne = fourState::bvAnd(rewriter, loc, notPoison, val);
      Value knownZero = fourState::bvAnd(rewriter, loc, notPoison, notVal);
      if (i == 0) { hasOne = knownOne; allZero = knownZero; orVals = val; }
      else {
        hasOne = fourState::bvOr(rewriter, loc, hasOne, knownOne);
        allZero = fourState::bvAnd(rewriter, loc, allZero, knownZero);
        orVals = fourState::bvOr(rewriter, loc, orVals, val);
      }
    }
    Value known = fourState::bvOr(rewriter, loc, hasOne, allZero);
    Value resPoison = fourState::bvNot(rewriter, loc, known);
    Value resVal = fourState::bvAnd(rewriter, loc, orVals, fourState::bvNot(rewriter, loc, resPoison));
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, resVal, resPoison));
    return success();
  }
};
struct FourStateXorOpConversion : OpConversionPattern<XorOp> {
  using OpConversionPattern<XorOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(XorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    if (w == 0 || adaptor.getInputs().empty()) return failure();
    if (adaptor.getInputs().size() == 1) { rewriter.replaceOp(op, adaptor.getInputs()[0]); return success(); }
    Value xorVals, orPoison;
    for (auto [i, inPair] : llvm::enumerate(adaptor.getInputs())) {
      auto [val, poison] = fourState::splitPair(rewriter, loc, inPair, w);
      if (i == 0) { xorVals = val; orPoison = poison; }
      else {
        xorVals = fourState::bvXor(rewriter, loc, xorVals, val);
        orPoison = fourState::bvOr(rewriter, loc, orPoison, poison);
      }
    }
    Value resVal = fourState::bvAnd(rewriter, loc, xorVals, fourState::bvNot(rewriter, loc, orPoison));
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, resVal, orPoison));
    return success();
  }
};
struct FourStateMuxOpConversion : OpConversionPattern<MuxOp> {
  using OpConversionPattern<MuxOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MuxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto intTy = dyn_cast<IntegerType>(op.getType());
    if (!intTy || intTy.getWidth() == 0)
      return rewriter.notifyMatchFailure(op, "only int mux supported in 4-state");
    unsigned w = intTy.getWidth();
    Value res = fourState::buildMux(rewriter, op.getLoc(), adaptor.getCond(), adaptor.getTrueValue(), adaptor.getFalseValue(), w);
    rewriter.replaceOp(op, res);
    return success();
  }
};
struct FourStateIcmpOpConversion : OpConversionPattern<ICmpOp> {
  using OpConversionPattern<ICmpOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ICmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getLhs().getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value zero1 = fourState::bvZero(rewriter, loc, 1);
    Value one1 = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    auto pred = op.getPredicate();
    if (pred == ICmpPredicate::ceq || pred == ICmpPredicate::cne) {
      Value eqVal = smt::EqOp::create(rewriter, loc, aVal, bVal);
      Value eqPoison = smt::EqOp::create(rewriter, loc, aPoison, bPoison);
      Value both = smt::AndOp::create(rewriter, loc, eqVal, eqPoison);
      Value valOut = (pred == ICmpPredicate::ceq)
          ? smt::IteOp::create(rewriter, loc, both, one1, zero1)
          : smt::IteOp::create(rewriter, loc, both, zero1, one1);
      rewriter.replaceOp(op, fourState::makePair(rewriter, loc, valOut, zero1));
      return success();
    }
    if (pred == ICmpPredicate::weq || pred == ICmpPredicate::wne) {
      Value notAP = fourState::bvNot(rewriter, loc, aPoison);
      Value notBP = fourState::bvNot(rewriter, loc, bPoison);
      Value diffVal = fourState::bvXor(rewriter, loc, aVal, bVal);
      Value mismatch = fourState::bvAnd(rewriter, loc, fourState::bvAnd(rewriter, loc, notAP, notBP), diffVal);
      Value zeroW = fourState::bvZero(rewriter, loc, w);
      Value isMismatch = smt::DistinctOp::create(rewriter, loc, mismatch, zeroW);
      Value valOut = (pred == ICmpPredicate::weq)
          ? smt::IteOp::create(rewriter, loc, isMismatch, zero1, one1)
          : smt::IteOp::create(rewriter, loc, isMismatch, one1, zero1);
      rewriter.replaceOp(op, fourState::makePair(rewriter, loc, valOut, zero1));
      return success();
    }
    Value aHasX = fourState::hasUnknown(rewriter, loc, aPoison);
    Value bHasX = fourState::hasUnknown(rewriter, loc, bPoison);
    Value hasX = smt::OrOp::create(rewriter, loc, aHasX, bHasX);
    Value poisOut = smt::IteOp::create(rewriter, loc, hasX, one1, zero1);
    Value cmpBool;
    if (pred == ICmpPredicate::eq) cmpBool = smt::EqOp::create(rewriter, loc, aVal, bVal);
    else if (pred == ICmpPredicate::ne) cmpBool = smt::DistinctOp::create(rewriter, loc, aVal, bVal);
    else {
      smt::BVCmpPredicate bvPred;
      switch (pred) {
      case ICmpPredicate::sge: bvPred = smt::BVCmpPredicate::sge; break;
      case ICmpPredicate::sgt: bvPred = smt::BVCmpPredicate::sgt; break;
      case ICmpPredicate::sle: bvPred = smt::BVCmpPredicate::sle; break;
      case ICmpPredicate::slt: bvPred = smt::BVCmpPredicate::slt; break;
      case ICmpPredicate::uge: bvPred = smt::BVCmpPredicate::uge; break;
      case ICmpPredicate::ugt: bvPred = smt::BVCmpPredicate::ugt; break;
      case ICmpPredicate::ule: bvPred = smt::BVCmpPredicate::ule; break;
      case ICmpPredicate::ult: bvPred = smt::BVCmpPredicate::ult; break;
      default: return rewriter.notifyMatchFailure(op, "bad predicate");
      }
      cmpBool = smt::BVCmpOp::create(rewriter, loc, bvPred, aVal, bVal);
    }
    Value cmpBV = smt::IteOp::create(rewriter, loc, cmpBool, one1, zero1);
    Value valOut = smt::IteOp::create(rewriter, loc, hasX, zero1, cmpBV);
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, valOut, poisOut));
    return success();
  }
};
struct FourStateParityOpConversion : OpConversionPattern<ParityOp> {
  using OpConversionPattern<ParityOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ParityOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getInput().getType()).getWidth();
    if (w == 0) return failure();
    auto [val, poison] = fourState::splitPair(rewriter, loc, adaptor.getInput(), w);
    Value hasX = fourState::hasUnknown(rewriter, loc, poison);
    auto *ctx = rewriter.getContext();
    Value runner = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), 0, val);
    for (unsigned i = 1; i < w; ++i) {
      Value ext = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), i, val);
      runner = smt::BVXOrOp::create(rewriter, loc, runner, ext);
    }
    Value zero1 = fourState::bvZero(rewriter, loc, 1);
    Value one1 = smt::BVConstantOp::create(rewriter, loc, 1, 1);
    Value poisOut = smt::IteOp::create(rewriter, loc, hasX, one1, zero1);
    Value valOut = smt::IteOp::create(rewriter, loc, hasX, zero1, runner);
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, valOut, poisOut));
    return success();
  }
};
struct FourStateReverseOpConversion : OpConversionPattern<ReverseOp> {
  using OpConversionPattern<ReverseOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ReverseOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getInput().getType()).getWidth();
    auto [val, poison] = fourState::splitPair(rewriter, loc, adaptor.getInput(), w);
    auto *ctx = rewriter.getContext();
    Value rVal = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), 0, val);
    Value rPoison = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), 0, poison);
    for (unsigned i = 1; i < w; ++i) {
      Value eVal = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), i, val);
      Value ePoison = smt::ExtractOp::create(rewriter, loc, fourState::bvTy(ctx, 1), i, poison);
      rVal = smt::ConcatOp::create(rewriter, loc, rVal, eVal);
      rPoison = smt::ConcatOp::create(rewriter, loc, rPoison, ePoison);
    }
    rewriter.replaceOp(op, fourState::makePair(rewriter, loc, rVal, rPoison));
    return success();
  }
};
struct FourStateAddOpConversion : OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    SmallVector<Value> vals, poisons;
    for (Value p : adaptor.getInputs()) {
      auto [v, poi] = fourState::splitPair(rewriter, loc, p, w);
      vals.push_back(v); poisons.push_back(poi);
    }
    Value orPoison = poisons[0];
    for (Value poi : llvm::drop_begin(poisons)) orPoison = fourState::bvOr(rewriter, loc, orPoison, poi);
    Value hasX = fourState::hasUnknown(rewriter, loc, orPoison);
    Value acc = vals[0];
    for (Value v : llvm::drop_begin(vals)) acc = smt::BVAddOp::create(rewriter, loc, acc, v);
    Value known = fourState::makeKnown(rewriter, loc, acc, w);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, hasX, allX, known);
    return success();
  }
};
struct FourStateMulOpConversion : OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    SmallVector<Value> vals, poisons;
    for (Value p : adaptor.getInputs()) {
      auto [v, poi] = fourState::splitPair(rewriter, loc, p, w);
      vals.push_back(v); poisons.push_back(poi);
    }
    Value orPoison = poisons[0];
    for (Value poi : llvm::drop_begin(poisons)) orPoison = fourState::bvOr(rewriter, loc, orPoison, poi);
    Value hasX = fourState::hasUnknown(rewriter, loc, orPoison);
    Value acc = vals[0];
    for (Value v : llvm::drop_begin(vals)) acc = smt::BVMulOp::create(rewriter, loc, acc, v);
    Value known = fourState::makeKnown(rewriter, loc, acc, w);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, hasX, allX, known);
    return success();
  }
};
struct FourStateSubOpConversion : OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(SubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value orPoison = fourState::bvOr(rewriter, loc, aPoison, bPoison);
    Value hasX = fourState::hasUnknown(rewriter, loc, orPoison);
    Value negB = smt::BVNegOp::create(rewriter, loc, bVal);
    Value resVal = smt::BVAddOp::create(rewriter, loc, aVal, negB);
    Value known = fourState::makeKnown(rewriter, loc, resVal, w);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, hasX, allX, known);
    return success();
  }
};
template <typename SourceOp, typename TargetOp> struct FourStateDivBase : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;
  using OpAdaptor = typename SourceOp::Adaptor;
  LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value orPoison = fourState::bvOr(rewriter, loc, aPoison, bPoison);
    Value hasX = fourState::hasUnknown(rewriter, loc, orPoison);
    auto resTy = this->typeConverter->convertType(op.getType());
    Value freeVar = smt::DeclareFunOp::create(rewriter, loc, resTy);
    Value zeroW = fourState::bvZero(rewriter, loc, w);
    Value isZero = smt::EqOp::create(rewriter, loc, bVal, zeroW);
    Value divVal = TargetOp::create(rewriter, loc, aVal, bVal);
    Value known = fourState::makeKnown(rewriter, loc, divVal, w);
    Value inner = smt::IteOp::create(rewriter, loc, isZero, freeVar, known);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, hasX, allX, inner);
    return success();
  }
};
struct FourStateShlOpConversion : OpConversionPattern<ShlOp> {
  using OpConversionPattern<ShlOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value amtHasX = fourState::hasUnknown(rewriter, loc, bPoison);
    Value vShift = smt::BVShlOp::create(rewriter, loc, aVal, bVal);
    Value pShift = smt::BVShlOp::create(rewriter, loc, aPoison, bVal);
    Value shifted = fourState::makePair(rewriter, loc, vShift, pShift);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, amtHasX, allX, shifted);
    return success();
  }
};
struct FourStateShrUOpConversion : OpConversionPattern<ShrUOp> {
  using OpConversionPattern<ShrUOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ShrUOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value amtHasX = fourState::hasUnknown(rewriter, loc, bPoison);
    Value vShift = smt::BVLShrOp::create(rewriter, loc, aVal, bVal);
    Value pShift = smt::BVLShrOp::create(rewriter, loc, aPoison, bVal);
    Value shifted = fourState::makePair(rewriter, loc, vShift, pShift);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, amtHasX, allX, shifted);
    return success();
  }
};
struct FourStateShrSOpConversion : OpConversionPattern<ShrSOp> {
  using OpConversionPattern<ShrSOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ShrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned w = cast<IntegerType>(op.getType()).getWidth();
    auto [aVal, aPoison] = fourState::splitPair(rewriter, loc, adaptor.getLhs(), w);
    auto [bVal, bPoison] = fourState::splitPair(rewriter, loc, adaptor.getRhs(), w);
    Value amtHasX = fourState::hasUnknown(rewriter, loc, bPoison);
    Value vShift = smt::BVAShrOp::create(rewriter, loc, aVal, bVal);
    Value pShift = smt::BVAShrOp::create(rewriter, loc, aPoison, bVal);
    Value shifted = fourState::makePair(rewriter, loc, vShift, pShift);
    Value allX = fourState::makeAllX(rewriter, loc, w);
    rewriter.replaceOpWithNewOp<smt::IteOp>(op, amtHasX, allX, shifted);
    return success();
  }
};
struct FourStateTruthTableOpConversion : OpConversionPattern<TruthTableOp> {
  using OpConversionPattern<TruthTableOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(TruthTableOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto lookup = op.getLookupTable();
    size_t n = adaptor.getInputs().size();
    if (n >= 31 || (1ULL << n) != lookup.size()) return failure();
    SmallVector<Value> cur;
    for (bool bit : lookup) {
      Value v = bit ? fourState::bvOnes(rewriter, loc, 1) : fourState::bvZero(rewriter, loc, 1);
      cur.push_back(fourState::makeKnown(rewriter, loc, v, 1));
    }
    for (int lvl = (int)n - 1; lvl >= 0; --lvl) {
      SmallVector<Value> nxt;
      for (size_t i = 0; i < cur.size(); i += 2) {
        Value c = adaptor.getInputs()[lvl];
        Value sel = fourState::buildMux(rewriter, loc, c, cur[i+1], cur[i], 1);
        nxt.push_back(sel);
      }
      cur = std::move(nxt);
    }
    rewriter.replaceOp(op, cur[0]);
    return success();
  }
};
} // end four-state patterns anon namespace

namespace {
struct ConvertCombToSMTPass
    : public circt::impl::ConvertCombToSMTBase<ConvertCombToSMTPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void circt::populateCombToSMTConversionPatterns(TypeConverter &converter,
                                                RewritePatternSet &patterns,
                                                bool fourState) {
  if (fourState) {
    patterns.add<FourStateReplicateOpConversion, FourStateExtractOpConversion,
                 FourStateConcatOpConversion, FourStateAndOpConversion,
                 FourStateOrOpConversion, FourStateXorOpConversion,
                 FourStateMuxOpConversion, FourStateIcmpOpConversion,
                 FourStateParityOpConversion, FourStateReverseOpConversion,
                 FourStateAddOpConversion, FourStateMulOpConversion,
                 FourStateSubOpConversion,
                 FourStateDivBase<DivSOp, smt::BVSDivOp>,
                 FourStateDivBase<DivUOp, smt::BVUDivOp>,
                 FourStateDivBase<ModSOp, smt::BVSRemOp>,
                 FourStateDivBase<ModUOp, smt::BVURemOp>,
                 FourStateShlOpConversion, FourStateShrUOpConversion,
                 FourStateShrSOpConversion,
                 FourStateTruthTableOpConversion>(converter,
                                                  patterns.getContext());
    return;
  }
  patterns.add<CombReplicateOpConversion, IcmpOpConversion, ExtractOpConversion,
               SubOpConversion, MuxOpConversion, ParityOpConversion,
               ReverseOpConversion, OneToOneOpConversion<ShlOp, smt::BVShlOp>,
               OneToOneOpConversion<ShrUOp, smt::BVLShrOp>,
               OneToOneOpConversion<ShrSOp, smt::BVAShrOp>,
               DivisionOpConversion<DivSOp, smt::BVSDivOp>,
               DivisionOpConversion<DivUOp, smt::BVUDivOp>,
               DivisionOpConversion<ModSOp, smt::BVSRemOp>,
               DivisionOpConversion<ModUOp, smt::BVURemOp>,
               VariadicToBinaryOpConversion<ConcatOp, smt::ConcatOp>,
               VariadicToBinaryOpConversion<AddOp, smt::BVAddOp>,
               VariadicToBinaryOpConversion<MulOp, smt::BVMulOp>,
               VariadicToBinaryOpConversion<AndOp, smt::BVAndOp>,
               VariadicToBinaryOpConversion<OrOp, smt::BVOrOp>,
               VariadicToBinaryOpConversion<XorOp, smt::BVXOrOp>>(
      converter, patterns.getContext());

  // TODO: there is one unsupported operation in the comb dialect:
  // 'truth_table'.
}

void ConvertCombToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<hw::HWDialect>();
  target.addIllegalOp<seq::FromClockOp>();
  target.addIllegalOp<seq::ToClockOp>();
  target.addIllegalDialect<comb::CombDialect>();
  target.addLegalDialect<smt::SMTDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter, fourState);
  // Also add HW patterns because some 'comb' canonicalizers produce constant
  // operations, i.e., even if there is absolutely no HW operation present
  // initially, we might have to convert one.
  populateHWToSMTConversionPatterns(converter, patterns, false, fourState);
  populateCombToSMTConversionPatterns(converter, patterns, fourState);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
