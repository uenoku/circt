//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs algebraic rewriting on Majority-Inverter Graphs (MIGs)
// for depth optimization using associativity and distributivity rules in
// majority-of-3 logic.
//
// The implementation (especially heuristics for pattern matching and algorithm
// structure) is heavily inspired by the MIG algebraic rewriting algorithm from
// the mockturtle library (https://github.com/lsils/mockturtle) and the paper
// "Majority-Inverter Graph: A New Paradigm for Logic Optimization" by Amarù et
// al. (2016).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synth-mig-algebraic-rewriting"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MIGALGEBRAICREWRITING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace {

/// Helper struct to represent ordered children of a majority operation
/// Uses PointerIntPair to store both the Value and its inversion flag

using ValueWithInversion = llvm::PointerIntPair<Value, 1, bool>;
struct OrderedChildren {
  // PointerIntPair stores Value pointer and inversion flag (1 bit)
  ValueWithInversion children[3];
  int64_t depths[3];

  OrderedChildren(synth::mig::MajorityInverterOp op,
                  llvm::SmallDenseMap<Value, int64_t> &depths) {
    auto operands = op.getInputs();
    assert(operands.size() == 3 && "Expected exactly 3 operands for majority");
    // Sort by depth (ascending - shallowest first)
    for (size_t i = 0; i < 3; ++i) {
      children[i].setPointer(operands[i]);
      children[i].setInt(op.isInverted(i));
    }

    std::sort(
        children, children + 3,
        [&depths](const ValueWithInversion &a, const ValueWithInversion &b) {
          return depths[a.getPointer()] < depths[b.getPointer()];
        });

    for (size_t i = 0; i < 3; ++i) {
      this->depths[i] = depths[children[i].getPointer()];
    }
  }

  static FailureOr<OrderedChildren>
  get(synth::mig::MajorityInverterOp op,
      IncrementalLongestPathAnalysis *analysis) {
    llvm::SmallDenseMap<Value, int64_t> depths;
    for (auto operand : op.getInputs()) {
      auto depth = analysis->getMaxDelay(operand, 0);
      if (failed(depth))
        return failure();
      depths[operand] = *depth;
    }
    return OrderedChildren(op, depths);
  }

  Value getValue(size_t idx) const {
    assert(idx < 3);
    return children[idx].getPointer();
  }

  void dump(llvm::raw_ostream &os) const {
    for (size_t i = 0; i < 3; ++i) {
      os << "  Child " << i << ": " << children[i].getPointer()
         << " (inverted: " << children[i].getInt() << ", depth: " << depths[i]
         << ")\n";
    }
  }

  bool isInverted(size_t idx) const {
    assert(idx < 3);
    return children[idx].getInt();
  }

  ValueWithInversion operator[](size_t idx) const {
    assert(idx < 3);
    return children[idx];
  }
};

/// Check if two values represent the same signal (considering inversions)
/// Returns true if they are the same signal, false otherwise
/// Also sets isInverted to true if one is the inversion of the other
bool isSameSignal(Value a, Value b, bool &isInverted) {
  isInverted = false;

  // Direct equality
  if (a == b)
    return true;

  if (auto constA = a.getDefiningOp<hw::ConstantOp>()) {
    if (auto constB = b.getDefiningOp<hw::ConstantOp>()) {
      if (constA.getValue() == constB.getValue())
        return true;
      if (constA.getValue() == ~constB.getValue()) {
        isInverted = true;
        return true;
      }
    }
  }

  // Check if one is an inverted version of the other through MIG operations
  auto checkInvertedMIG = [](Value val, Value target) -> bool {
    if (auto migOp = val.getDefiningOp<synth::mig::MajorityInverterOp>()) {
      // Check for single-input inverted MIG (acts as NOT gate)
      if (migOp.getNumOperands() == 1 && migOp.isInverted(0) &&
          migOp.getOperand(0) == target)
        return true;
    }
    return false;
  };

  if (checkInvertedMIG(a, b) || checkInvertedMIG(b, a)) {
    isInverted = true;
    return true;
  }

  return false;
}

/// Overload for backward compatibility
bool isSameSignal(Value a, Value b) {
  bool isInverted;
  return isSameSignal(a, b, isInverted);
}

static Value createMajorityFunction(OpBuilder &rewriter, Location loc, Value a,
                                    Value b, Value c, bool invertA,
                                    bool invertB, bool invertC) {
  std::array<Value, 3> inputs = {a, b, c};
  std::array<bool, 3> inverts = {invertA, invertB, invertC};
  // for (size_t i = 0; i < 3; ++i) {
  //   if (inverts[i] && inputs[i].getDefiningOp<hw::ConstantOp>()) {
  //     inputs[i] = rewriter.create<hw::ConstantOp>(
  //         loc, ~inputs[i].getDefiningOp<hw::ConstantOp>().getValue());
  //     inverts[i] = false;
  //   }
  // }
  return rewriter.createOrFold<synth::mig::MajorityInverterOp>(loc, inputs,
                                                               inverts);
}

static Value createMajorityFunction(OpBuilder &rewriter, Location loc,
                                    Value input, bool invert) {
  return createMajorityFunction(rewriter, loc, input, input, input, invert,
                                invert, invert);
}

/// Enhanced associativity rewrite with depth checking and inversion handling
static FailureOr<Value> tryAssociativityRewriteWithDepth(
    synth::mig::MajorityInverterOp topOp, const OrderedChildren &topChildren,
    PatternRewriter &rewriter, IncrementalLongestPathAnalysis *depthAnalysis) {

  // Check if the last (highest level) child is also a majority operation
  auto childOp =
      topChildren.getValue(2).getDefiningOp<synth::mig::MajorityInverterOp>();
  LDBG() << "Checking associativity rewrite for " << topOp << "\n";
  LDBG() << "  Child: " << childOp << "\n";
  if (!childOp || topChildren.isInverted(2))
    return Value();

  // Get children of the child operation
  auto childChildren = OrderedChildren::get(childOp, depthAnalysis);
  if (failed(childChildren))
    return rewriter.notifyMatchFailure(topOp, "Failed to get child depth");

  // Look for associativity opportunities: maj(v, w, maj(x, y, z))
  Value v = topChildren.getValue(0);
  Value w = topChildren.getValue(1);
  Value x = childChildren->getValue(0);
  Value y = childChildren->getValue(1);
  Value z = childChildren->getValue(2);

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      auto v = topChildren[i];
      auto x = (*childChildren)[j];
      bool invert = false;
      if (!isSameSignal(v.getPointer(), x.getPointer(), invert))
        continue;
      invert ^= v.getInt() ^ x.getInt();
      LDBG() << "  Found common signal: " << v.getPointer()
             << " (invert: " << invert << ")\n";

      auto w = topChildren[1 - i];
      // Ok try swapping w and y or z
      auto y = (*childChildren)[2 - j];
      auto z = (*childChildren)[2];

      if (invert) {
        // Completely associativity.
        // M(v, w, M(v', y, z)) -> M(v, w, M(w, y, z))
        // Benefits when w == y
        bool isInverted;
        if (isSameSignal(w.getPointer(), y.getPointer(), isInverted)) {
          isInverted ^= w.getInt() ^ y.getInt();
          LDBG() << "  Found associativity rewrite: " << topOp << "\n";
          if (isInverted) {
            // M(v, w, M(w, w', z)) -> M(v, w, z)
            return createMajorityFunction(
                rewriter, topOp.getLoc(), v.getPointer(), w.getPointer(),
                z.getPointer(), v.getInt(), w.getInt(), z.getInt());
          }
          // M(v, w, M(w, w, z)) -> M(v, w, w)  -> w
          if (w.getInt())
            return createMajorityFunction(rewriter, topOp.getLoc(),
                                          w.getPointer(), true);
          return w.getPointer();
        }
      } else {
        // Assoc
        // M(v, w, M(v, y, z)) = M(z, w, M(v, y, w))
        int64_t depthZ = childChildren->depths[2];
        int64_t depthY = childChildren->depths[1];
        if (depthZ != depthY) {
          // M(v, y, w)
          auto newOp = createMajorityFunction(
              rewriter, topOp.getLoc(), v.getPointer(), y.getPointer(),
              w.getPointer(), v.getInt(), y.getInt(), w.getInt());
          // M(v, w, M(v, y, w))
          auto finalOp = createMajorityFunction(
              rewriter, topOp.getLoc(), z.getPointer(), w.getPointer(), newOp,
              z.getInt(), w.getInt(), false);
          return finalOp;
        }
      }
    }

    /*
    LDBG() << "  Child children:\n";
    topChildren.dump(llvm::errs());
    childChildren->dump(llvm::errs());


    // Helper to create rewrite with proper inversion handling
    auto tryRewrite = [&](Value common, Value other, Value childOther,
                          Value remaining, bool vIsCommon,
                          bool xIsCommon) -> Value {
      bool isInverted = false;
      if (!isSameSignal(common, xIsCommon ? x : y, isInverted))
        return nullptr;

      // Handle inversion flags properly
      SmallVector<bool> innerInversions = {
          topOp.isInverted(vIsCommon ? 0 : 1),
          topOp.isInverted(vIsCommon ? 1 : 0),
          static_cast<bool>(childOp.isInverted(xIsCommon ? 1 : 0) ^
    isInverted)};

      auto innerMaj = createMajorityFunction(
          rewriter, topOp.getLoc(), common, other, childOther,
    innerInversions[0], innerInversions[1], innerInversions[2]);

      SmallVector<bool> outerInversions = {
          static_cast<bool>(childOp.isInverted(2) ^ isInverted), false,
          topOp.isInverted(vIsCommon ? 0 : 1)};

      auto newOp = createMajorityFunction(rewriter, topOp.getLoc(), remaining,
                                          innerMaj, common, outerInversions[0],
                                          outerInversions[1],
    outerInversions[2]);

      return newOp;
    };

    // Try different patterns
    if (auto result = tryRewrite(v, w, y, z, true, true))
      return result;
    if (auto result = tryRewrite(v, w, x, z, true, false))
      return result;
    if (auto result = tryRewrite(w, v, y, z, false, true))
      return result;
    if (auto result = tryRewrite(w, v, x, z, false, false))
      return result;
      */
  }

  return Value();
}

/// Pattern to rewrite MIG operations for depth reduction
struct MIGDepthReductionPattern
    : public OpRewritePattern<synth::mig::MajorityInverterOp> {
  bool allowAreaIncrease;
  IncrementalLongestPathAnalysis *depthAnalysis;

  MIGDepthReductionPattern(MLIRContext *context, bool allowAreaIncrease,
                           IncrementalLongestPathAnalysis *analysis)
      : OpRewritePattern<synth::mig::MajorityInverterOp>(context,
                                                         /*benefit=*/1),
        allowAreaIncrease(allowAreaIncrease), depthAnalysis(analysis) {}

  LogicalResult matchAndRewrite(synth::mig::MajorityInverterOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle 3-input majority operations for now
    if (op.getNumOperands() != 3)
      return failure();

    // Get current depth of the operation
    // Get ordered children (sorted by level) with depth analysis
    auto orderedChildren = OrderedChildren::get(op, depthAnalysis);
    if (failed(orderedChildren))
      return rewriter.notifyMatchFailure(op, "Failed to get ordered children");

    if (orderedChildren->depths[2] <= orderedChildren->depths[1] + 1)
      return failure();
    LDBG() << "Found candidate:\n" << op << "\n";
    for (auto [child, depth] :
         llvm::zip(orderedChildren->children, orderedChildren->depths)) {
      LDBG() << "  Child: " << child.getPointer()
             << " (inverted: " << child.getInt() << ", depth: " << depth
             << ")\n";
    }

    // Check if the last child is a majority operation with higher level
    auto childOp = orderedChildren->getValue(2)
                       .getDefiningOp<synth::mig::MajorityInverterOp>();
    if (!childOp)
      return failure();

    // For single fanout constraint (when area increase is not allowed)
    if (!allowAreaIncrease && !childOp.getResult().hasOneUse())
      return failure();

    // Try associativity rewrite first with depth checking
    auto newOp = tryAssociativityRewriteWithDepth(op, *orderedChildren,
                                                  rewriter, depthAnalysis);
    if (failed(newOp))
      return rewriter.notifyMatchFailure(op, "Failed associativity rewrite");

    if (succeeded(newOp) && *newOp) {
      // Mark the new operation to prevent infinite loops
      LDBG() << "  Found associativity rewrite: " << op << " -> " << *newOp
             << "\n";
      rewriter.replaceOp(op, *newOp);
      return success();
    }

    // Try distributivity rewrite if area increase is allowed
    if (false && allowAreaIncrease) {
      auto childChildrenFailureOr =
          OrderedChildren::get(childOp, depthAnalysis);
      if (failed(childChildrenFailureOr))
        return rewriter.notifyMatchFailure(op, "Failed to get child depth");
      auto &childChildren = *childChildrenFailureOr;

      // maj(v, w, maj(x, y, z)) -> maj(maj(v, w, x), maj(v, w, y), z)
      // Handle inversion flags properly
      SmallVector<bool> leftInversions = {orderedChildren->isInverted(0),
                                          orderedChildren->isInverted(1),
                                          childChildren.isInverted(0)};
      SmallVector<bool> rightInversions = {orderedChildren->isInverted(0),
                                           orderedChildren->isInverted(1),
                                           childChildren.isInverted(1)};

      auto leftMaj = rewriter.create<synth::mig::MajorityInverterOp>(
          op.getLoc(), op.getType(),
          ValueRange{orderedChildren->getValue(0), orderedChildren->getValue(1),
                     childChildren.getValue(0)},
          rewriter.getDenseBoolArrayAttr(leftInversions));

      auto rightMaj = rewriter.create<synth::mig::MajorityInverterOp>(
          op.getLoc(), op.getType(),
          ValueRange{orderedChildren->getValue(0), orderedChildren->getValue(1),
                     childChildren.getValue(1)},
          rewriter.getDenseBoolArrayAttr(rightInversions));

      auto newOp = rewriter.create<synth::mig::MajorityInverterOp>(
          op.getLoc(), op.getType(),
          ValueRange{leftMaj.getResult(), rightMaj.getResult(),
                     childChildren.getValue(2)},
          rewriter.getDenseBoolArrayAttr(
              {false, false, childChildren.isInverted(2)}));

      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
  }
};

/// The main MIG algebraic rewriting pass
struct MIGAlgebraicRewritingPass
    : public circt::synth::impl::MIGAlgebraicRewritingBase<
          MIGAlgebraicRewritingPass> {

  using circt::synth::impl::MIGAlgebraicRewritingBase<
      MIGAlgebraicRewritingPass>::MIGAlgebraicRewritingBase;

  void runOnOperation() override {
    auto module = getOperation();
    mlir::AnalysisManager am = getAnalysisManager();

    // Create longest path analysis for depth calculation
    IncrementalLongestPathAnalysis depthAnalysis(module, am);

    RewritePatternSet patterns(&getContext());

    patterns.add<MIGDepthReductionPattern>(&getContext(), allowAreaIncrease,
                                           &depthAnalysis);

    auto config = GreedyRewriteConfig();
    config.setListener(&depthAnalysis).setUseTopDownTraversal(true);

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
