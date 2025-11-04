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

/// Minato-Morreale ISOP algorithm.
/// The implementation is heavily inspired by the implementation in mockturtle.
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

  if (varIndex == numVars) {
    // No more variables, shouldn't happen if algorithm is correct
    return;
  }

  // Compute positive and negative cofactors
  APInt on1 = computeCofactor(on, numVars, varIndex, true);
  APInt on0 = computeCofactor(on, numVars, varIndex, false);

  // Compute the intersection (where both cofactors are true)
  APInt onAnd = on1 & on0;

  // Recurse on the common part (variable doesn't matter)
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
static Value buildAnd(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                      ArrayRef<bool> inverted) {
  assert(values.size() == inverted.size() && "Size mismatch");

  if (values.empty())
    return {};

  if (values.size() == 1)
    return aig::AndInverterOp::create(builder, loc, values[0], inverted[0]);

  // Use variadic and_inv
  return aig::AndInverterOp::create(builder, loc, values, inverted);
}

/// Build an OR operation from a list of values.
/// In AIG, OR is implemented as NOT(AND(NOT a, NOT b, ...))
/// TODO: Generalize to MIG etc.
static Value buildOr(OpBuilder &builder, Location loc, ArrayRef<Value> values) {
  if (values.empty())
    return {};

  if (values.size() == 1)
    return values[0];

  // Use De Morgan's law: a OR b OR c = NOT(NOT a AND NOT b AND NOT c)
  SmallVector<bool> inverted(values.size(), true);
  auto andOp = aig::AndInverterOp::create(builder, loc, values, inverted);
  // Invert the result
  return aig::AndInverterOp::create(builder, loc, andOp, true);
}

/// Build a balanced SOP structure from the SOP form.
static Value buildBalancedSOP(OpBuilder &builder, Location loc,
                              const SOPForm &sop, ArrayRef<Value> inputs) {
  SmallVector<Value> productTerms;

  // Build each product term (cube)
  for (const auto &cube : sop.cubes) {
    SmallVector<Value> literals;
    SmallVector<bool> literalInverted;

    for (unsigned i = 0; i < sop.numVars; ++i) {
      if (cube.mask[i]) {
        literals.push_back(inputs[i]);
        literalInverted.push_back(cube.inverted[i]);
      }
    }

    if (literals.empty())
      continue;

    // Build AND for this product term
    Value product = buildAnd(builder, loc, literals, literalInverted);
    if (product)
      productTerms.push_back(product);
  }

  assert(!productTerms.empty() && "No product terms");

  // Build OR of all product terms
  return buildOr(builder, loc, productTerms);
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Compute the delay from each input to the output in a SOP structure.
/// Uses balanced tree depth model (ceil(log2(n))) for accurate delay
/// estimation.
static void computeSOPDelays(const SOPForm &sop, unsigned numInputs,
                             SmallVectorImpl<DelayType> &delays) {
  delays.resize(numInputs, 0);

  // For each input, find the maximum depth it appears at in the SOP structure
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
  }
}

/// Pattern that performs SOP balancing on cuts.
struct SOPBalancingPattern : public CutRewritePattern {
  // Temporary storage for computed delays
  mutable SmallVector<DelayType> tempDelays;

  SOPBalancingPattern(MLIRContext *context) : CutRewritePattern(context) {}

  bool match(const Cut &cut, MatchResult &result) const override {
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
    // TODO: Pass arrival times of inputs.
    computeSOPDelays(sop, cut.getInputSize(), tempDelays);
    result.delays = tempDelays; // ArrayRef points to tempDelays

    return true;
  }

  FailureOr<Operation *> rewrite(OpBuilder &builder, Cut &cut) const override {
    // Get the truth table for the cut
    const auto &tt = cut.getTruthTable();

    // Extract SOP form from truth table
    SOPForm sop = extractSOPFromTruthTable(tt);

    // Build balanced SOP structure
    Value result = buildBalancedSOP(builder, cut.getRoot()->getLoc(), sop,
                                    cut.inputs.getArrayRef());

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
