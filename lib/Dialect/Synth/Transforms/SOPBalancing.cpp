//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements SOP (Sum-of-Products) balancing for delay optimization.
// The algorithm is based on "Delay Optimization Using SOP Balancing" by
// Mishchenko et al. (ICCAD 2011).
//
// SOP balancing restructures logic networks by:
// 1. Deriving ISOP (sum-of-products) representations for cuts using
//    Minato-Morreale ISOP algorithm. The implementation is heavily inspired by
//    the implementation in mockturtle.
// 2. Balancing the SOP to minimize delay
// 3. Rewriting the AIG with the balanced structure
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/NPNClass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "synth-sop-balancing"

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_SOPBALANCING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt::synth;

namespace {

//===----------------------------------------------------------------------===//
// SOP Representation and Balancing
//===----------------------------------------------------------------------===//

/// Represents a product term (cube) in a sum-of-products expression.
/// Each cube is a conjunction of literals (variables or their negations).
struct Cube {
  // Bitmask indicating which variables appear in this cube
  llvm::APInt mask;
  // Bitmask indicating which variables are negated
  llvm::APInt inverted;

  Cube(unsigned numVars) : mask(numVars, 0), inverted(numVars, 0) {}

  unsigned size() const { return mask.popcount(); }
};

/// Represents a sum-of-products expression.
struct SOPForm {
  SmallVector<Cube> cubes;
  unsigned numVars;

  SOPForm(unsigned numVars) : numVars(numVars) {}
};

/// Compute cofactor: f_xi (positive) or f_!xi (negative).
static APInt computeCofactor(const APInt &f, unsigned numVars, unsigned var,
                             bool positive) {
  uint32_t numBits = 1u << numVars;
  APInt result(numBits, 0);

  uint32_t shift = 1u << var;
  for (uint32_t i = 0; i < numBits; ++i) {
    if (positive) {
      // Positive cofactor: set bit i if bit (i | shift) is set in f
      if (f[i | shift])
        result.setBit(i);
    } else {
      // Negative cofactor: set bit i if bit (i & ~shift) is set in f
      if (f[i & ~shift])
        result.setBit(i);
    }
  }

  return result;
}

/// Minato-Morreale ISOP algorithm (See "Finding All Simple Disjunctive
/// Decompositions Using Irredundant Sum-of-Products Forms"Sec 3.2). The
/// implementation is heavily inspired by the implementation in mockturtle.
static void isopRecursive(const APInt &on, unsigned numVars, unsigned varIndex,
                          const Cube &cube, SOPForm &result) {
  // Terminal case: nothing to cover
  if (on.isZero())
    return;

  // Terminal case: all bits set, add the cube
  if (on.isAllOnes()) {
    result.cubes.push_back(cube);
    return;
  }

  assert(varIndex != numVars && "Ran out of variables");

  // Compute positive and negative cofactors
  APInt on1 = computeCofactor(on, numVars, varIndex, true);
  APInt on0 = computeCofactor(on, numVars, varIndex, false);

  // Compute the intersection where both cofactors are true
  APInt onAnd = on1 & on0;

  // Recurse on the common part
  isopRecursive(onAnd, numVars, varIndex + 1, cube, result);

  // Compute the differences
  APInt onDiff1 = on1 & ~onAnd;
  APInt onDiff0 = on0 & ~onAnd;

  // Recurse with variable = 1 (positive literal)
  if (!onDiff1.isZero()) {
    Cube cube1 = cube;
    cube1.mask.setBit(varIndex);
    // Not inverted (positive literal)
    isopRecursive(onDiff1, numVars, varIndex + 1, cube1, result);
  }

  // Recurse with variable = 0 (negative literal)
  if (!onDiff0.isZero()) {
    Cube cube0 = cube;
    cube0.mask.setBit(varIndex);
    cube0.inverted.setBit(varIndex); // Inverted (negative literal)
    isopRecursive(onDiff0, numVars, varIndex + 1, cube0, result);
  }
}

/// Extract ISOP (Irredundant Sum-of-Products) from truth table.
/// Uses Minato-Morreale algorithm for efficient ISOP computation.
static SOPForm extractSOPFromTruthTable(const BinaryTruthTable &tt) {
  SOPForm sop(tt.numInputs);

  if (tt.numInputs == 0 || tt.table.isZero())
    return sop;

  // Start recursive ISOP extraction
  Cube emptyCube(tt.numInputs);
  isopRecursive(tt.table, tt.numInputs, 0, emptyCube, sop);

  return sop;
}

/// Build an AND operation from a list of values using variadic and_inv.
/// Builds a balanced tree based on arrival times to minimize delay.
static Value buildAnd(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                      ArrayRef<bool> inverted,
                      ArrayRef<DelayType> arrivalTimes = {}) {
  assert(values.size() == inverted.size() && "Size mismatch");
  assert(arrivalTimes.empty() ||
         values.size() == arrivalTimes.size() && "Arrival times size mismatch");

  if (values.empty())
    return {};

  if (values.size() == 1)
    return aig::AndInverterOp::create(builder, loc, values[0], inverted[0]);

  // If no arrival times provided, use variadic and_inv
  if (arrivalTimes.empty())
    return aig::AndInverterOp::create(builder, loc, values, inverted);

  // Build balanced tree based on arrival times
  // Strategy: pair inputs with similar arrival times to minimize critical path
  SmallVector<std::tuple<Value, bool, DelayType>> items;
  for (unsigned i = 0; i < values.size(); ++i)
    items.push_back({values[i], inverted[i], arrivalTimes[i]});

  // Sort by arrival time (descending) so we process late-arriving signals first
  llvm::sort(items, [](const auto &a, const auto &b) {
    return std::get<2>(a) > std::get<2>(b);
  });

  // Build balanced binary tree
  while (items.size() > 1) {
    SmallVector<std::tuple<Value, bool, DelayType>> nextLevel;

    for (unsigned i = 0; i + 1 < items.size(); i += 2) {
      auto [val1, inv1, time1] = items[i];
      auto [val2, inv2, time2] = items[i + 1];

      // Create AND of the two values
      SmallVector<Value, 2> andInputs = {val1, val2};
      SmallVector<bool, 2> andInverted = {inv1, inv2};
      Value andResult =
          aig::AndInverterOp::create(builder, loc, andInputs, andInverted);

      // New arrival time is max of inputs + 1 gate delay
      DelayType newTime = std::max(time1, time2) + 1;
      nextLevel.push_back({andResult, false, newTime});
    }

    // Handle odd element
    if (items.size() % 2 == 1)
      nextLevel.push_back(items.back());

    items = std::move(nextLevel);
  }

  auto [finalVal, finalInv, finalTime] = items[0];
  if (finalInv)
    return aig::AndInverterOp::create(builder, loc, finalVal, true);
  return finalVal;
}

/// Build an OR operation from a list of values.
/// In AIG, OR is implemented as NOT(AND(NOT a, NOT b, ...))
/// Builds a balanced tree based on arrival times to minimize delay.
static Value buildOr(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                     ArrayRef<DelayType> arrivalTimes = {}) {
  if (values.empty())
    return {};

  if (values.size() == 1)
    return values[0];

  // OR(a, b, ...) = NOT(AND(NOT a, NOT b, ...))
  // Build the AND with all inputs inverted, then invert the result
  SmallVector<bool> inverted(values.size(), true);

  if (arrivalTimes.empty()) {
    auto andOp = aig::AndInverterOp::create(builder, loc, values, inverted);
    return aig::AndInverterOp::create(builder, loc, andOp, true);
  }

  // Build balanced AND tree with arrival times
  auto andOp = buildAnd(builder, loc, values, inverted, arrivalTimes);
  // Invert the result
  return aig::AndInverterOp::create(builder, loc, andOp, true);
}

/// Build a balanced SOP structure from the SOP form.
/// Uses input arrival times to build a delay-optimized structure.
static Value buildBalancedSOP(OpBuilder &builder, Location loc,
                              const SOPForm &sop, ArrayRef<Value> inputs,
                              ArrayRef<DelayType> inputArrivalTimes = {}) {
  assert(inputArrivalTimes.empty() ||
         inputs.size() == inputArrivalTimes.size() &&
             "Input arrival times size mismatch");

  SmallVector<Value> productTerms;
  SmallVector<DelayType> productArrivalTimes;

  // Build each product term (cube)
  for (const auto &cube : sop.cubes) {
    SmallVector<Value> literals;
    SmallVector<bool> literalInverted;
    SmallVector<DelayType> literalArrivalTimes;

    for (unsigned i = 0; i < sop.numVars; ++i) {
      if (cube.mask[i]) {
        literals.push_back(inputs[i]);
        literalInverted.push_back(cube.inverted[i]);
        if (!inputArrivalTimes.empty())
          literalArrivalTimes.push_back(inputArrivalTimes[i]);
      }
    }

    if (literals.empty())
      continue;

    // Build AND for this product term with arrival time awareness
    Value product =
        buildAnd(builder, loc, literals, literalInverted, literalArrivalTimes);
    if (product) {
      productTerms.push_back(product);

      // Compute arrival time for this product term
      if (!inputArrivalTimes.empty()) {
        DelayType maxInputTime = 0;
        for (auto time : literalArrivalTimes)
          maxInputTime = std::max(maxInputTime, time);

        // Add delay for the AND tree
        DelayType andDepth =
            literals.size() > 1 ? llvm::Log2_32_Ceil(literals.size()) : 0;
        productArrivalTimes.push_back(maxInputTime + andDepth);
      }
    }
  }

  assert(!productTerms.empty() && "No product terms");

  // Build OR of all product terms with arrival time awareness
  return buildOr(builder, loc, productTerms, productArrivalTimes);
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Compute the delay from each input to the output in a SOP structure.
/// Uses balanced tree depth model (ceil(log2(n))) for accurate delay
/// estimation. Takes input arrival times into account.
static DelayType computeSOPDelays(const SOPForm &sop, unsigned numInputs,
                                  ArrayRef<DelayType> inputArrivalTimes,
                                  SmallVectorImpl<DelayType> &delays) {
  delays.resize(numInputs, 0);

  DelayType maxOutputArrivalTime = 0;

  // For each input, compute the delay contribution to the output
  for (unsigned inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    DelayType maxDelay = 0;

    // Check each product term (cube) that uses this input
    for (const auto &cube : sop.cubes) {
      if (!cube.mask[inputIdx])
        continue; // This cube doesn't use this input

      // Compute the depth of the AND tree for this cube
      // Balanced binary tree depth is ceil(log2(size))
      unsigned cubeSize = cube.size();
      DelayType andDepth = cubeSize > 1 ? llvm::Log2_32_Ceil(cubeSize) : 0;

      // Add depth for the OR gate at the top (if there are multiple cubes)
      DelayType totalDelay = andDepth;
      if (sop.cubes.size() > 1) {
        // Compute OR tree depth
        DelayType orDepth = llvm::Log2_32_Ceil(sop.cubes.size());
        totalDelay += orDepth;
      }

      maxDelay = std::max(maxDelay, totalDelay);
    }

    delays[inputIdx] = maxDelay;

    // Compute actual output arrival time for this input path
    DelayType outputArrivalTime = inputArrivalTimes[inputIdx] + maxDelay;
    maxOutputArrivalTime = std::max(maxOutputArrivalTime, outputArrivalTime);
  }

  return maxOutputArrivalTime;
}

/// Pattern that performs SOP balancing on cuts.
struct SOPBalancingPattern : public CutRewritePattern {
  // Temporary storage for computed delays
  mutable SmallVector<DelayType> tempDelays;

  SOPBalancingPattern(MLIRContext *context) : CutRewritePattern(context) {}

  bool match(const Cut &cut, CutEnumerator &enumerator,
             MatchResult &result) const override {
    // Match any non-trivial cut with a single output
    if (cut.isTrivialCut() || cut.getOutputSize() != 1)
      return false;

    // Get the truth table for the cut
    const auto &tt = cut.getTruthTable();

    // Extract SOP form from truth table
    // TODO: mockturtle nicely parametrizes sop/bi-decomposition/aiker etc as
    // "resynthesis" engine. Consider taking similar approach if we want to do
    // more than SOP.
    SOPForm sop = extractSOPFromTruthTable(tt);

    // If SOP is empty, don't match
    if (sop.cubes.empty())
      return false;

    // Compute area (number of gates in the balanced structure)
    // This is a rough estimate: sum of cube sizes + OR gates
    unsigned totalGates = 0;
    for (const auto &cube : sop.cubes) {
      if (cube.size() > 1)
        totalGates += cube.size() - 1; // AND gates in this cube
    }
    if (sop.cubes.size() > 1)
      totalGates += sop.cubes.size() - 1; // OR gates

    result.area = static_cast<double>(totalGates);

    // Compute delays from each input to the output.
    SmallVector<DelayType, 6> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes)))
      return false;

    // Compute delays using input arrival times
    computeSOPDelays(sop, cut.getInputSize(), arrivalTimes, tempDelays);
    result.delays = tempDelays; // ArrayRef points to tempDelays

    return true;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, CutEnumerator &enumerator,
                                 Cut &cut) const override {
    // Get the truth table for the cut
    const auto &tt = cut.getTruthTable();

    // Extract SOP form from truth table
    SOPForm sop = extractSOPFromTruthTable(tt);

    // Get input arrival times for delay-optimized balancing
    SmallVector<DelayType, 6> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes))) {
      // Fall back to building without arrival time information
      Value result = buildBalancedSOP(builder, cut.getRoot()->getLoc(), sop,
                                      cut.inputs.getArrayRef());
      auto *op = result.getDefiningOp();
      if (!op) {
        op =
            aig::AndInverterOp::create(builder, cut.getRoot()->getLoc(), result,
                                       /*invert=*/false);
      }
      return op;
    }

    // Build balanced SOP structure with arrival time awareness
    Value result = buildBalancedSOP(builder, cut.getRoot()->getLoc(), sop,
                                    cut.inputs.getArrayRef(), arrivalTimes);

    auto *op = result.getDefiningOp();
    if (!op) {
      // Create an unary and inverter op if the result is not an operation.
      op = aig::AndInverterOp::create(builder, cut.getRoot()->getLoc(), result,
                                      /*invert=*/false);
    }

    return op;
  }

  unsigned getNumOutputs() const override { return 1; }

  StringRef getPatternName() const override { return "sop-balancing"; }
};

//===----------------------------------------------------------------------===//
// SOP Balancing Pass
//===----------------------------------------------------------------------===//

struct SOPBalancingPass
    : public circt::synth::impl::SOPBalancingBase<SOPBalancingPass> {
  using SOPBalancingBase::SOPBalancingBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Create cut rewriter options
    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = true; // Allow nodes without matches to pass through

    // Create SOP balancing pattern
    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    patterns.push_back(
        std::make_unique<SOPBalancingPattern>(module->getContext()));

    // Create pattern set
    CutRewritePatternSet patternSet(std::move(patterns));

    // Create and run cut rewriter
    CutRewriter rewriter(options, patternSet);
    if (failed(rewriter.run(module)))
      return signalPassFailure();
  }
};

} // namespace
