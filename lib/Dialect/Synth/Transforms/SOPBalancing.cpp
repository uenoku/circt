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
  void dump(llvm::raw_ostream &os) const {
    os << "SOPForm: " << numVars << " vars, " << cubes.size() << " cubes\n";
    for (const auto &cube : cubes) {
      os << "  (";
      for (unsigned i = 0; i < numVars; ++i) {
        if (cube.mask[i]) {
          os << (cube.inverted[i] ? "!" : "");
          os << "x" << i << " ";
        }
      }
      os << ")\n";
    }
  }
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
/// Uses a priority queue to greedily pair nodes with earliest arrival times.
static Value buildAnd(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                      ArrayRef<bool> inverted,
                      ArrayRef<DelayType> arrivalTimes) {
  assert(values.size() == inverted.size() && "Size mismatch");
  assert(values.size() == arrivalTimes.size() && "Arrival times size mismatch");

  if (values.empty())
    return {};

  if (values.size() == 1)
    return aig::AndInverterOp::create(builder, loc, values[0], inverted[0]);

  // Build balanced tree based on arrival times using priority queue
  // Strategy: greedily pair nodes with earliest arrival times to minimize delay
  SmallVector<ValueWithArrivalTime> nodes;
  size_t valueNumber = 0;
  for (unsigned i = 0; i < values.size(); ++i)
    nodes.push_back(ValueWithArrivalTime(values[i], arrivalTimes[i],
                                         inverted[i], valueNumber++));

  ValueWithArrivalTime result =
      buildBalancedTreeWithArrivalTimes<ValueWithArrivalTime>(
          nodes,
          // Combine two nodes
          [&](const ValueWithArrivalTime &node1,
              const ValueWithArrivalTime &node2) {
            Value andResult = aig::AndInverterOp::create(
                builder, loc, node1.getValue(), node2.getValue(),
                node1.isInverted(), node2.isInverted());

            // New arrival time is max of inputs + 1 gate delay
            DelayType newTime =
                std::max(node1.getArrivalTime(), node2.getArrivalTime()) + 1;
            return ValueWithArrivalTime(andResult, newTime, false,
                                        valueNumber++);
          });

  if (result.isInverted())
    return aig::AndInverterOp::create(builder, loc, result.getValue(), true);
  return result.getValue();
}

/// Build an OR operation from a list of values.
/// In AIG, OR is implemented as NOT(AND(NOT a, NOT b, ...))
/// Builds a balanced tree based on arrival times to minimize delay.
static Value buildOr(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                     ArrayRef<DelayType> arrivalTimes) {
  if (values.empty())
    return {};

  if (values.size() == 1)
    return values[0];

  // OR(a, b, ...) = NOT(AND(NOT a, NOT b, ...))
  // Build the AND with all inputs inverted, then invert the result
  SmallVector<bool> inverted(values.size(), true);
  auto andOp = buildAnd(builder, loc, values, inverted, arrivalTimes);
  // Invert the result
  return aig::AndInverterOp::create(builder, loc, andOp, true);
}

/// Simulate building a balanced AND tree and return the output arrival time.
/// This matches the logic in buildAnd() but only computes timing.
/// Uses a priority queue to greedily pair nodes with earliest arrival times.
static DelayType simulateAndTree(ArrayRef<DelayType> inputArrivalTimes) {
  if (inputArrivalTimes.empty())
    return 0;
  return buildBalancedTreeWithArrivalTimes<DelayType>(
      inputArrivalTimes,
      // Combine: max of two delays + 1 gate delay
      [&](auto node1, auto node2) { return std::max(node1, node2) + 1; });
}

/// Build a balanced SOP structure from the SOP form.
/// Uses input arrival times to build a delay-optimized structure.
static Value buildBalancedSOP(OpBuilder &builder, Location loc,
                              const SOPForm &sop, ArrayRef<Value> inputs,
                              ArrayRef<DelayType> inputArrivalTimes) {
  assert(inputs.size() == inputArrivalTimes.size() &&
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
        literalArrivalTimes.push_back(inputArrivalTimes[i]);
      }
    }

    if (literals.empty())
      continue;

    // Build AND for this product term with arrival time awareness
    Value product =
        buildAnd(builder, loc, literals, literalInverted, literalArrivalTimes);
    if (product)
      productTerms.push_back(product);
  }

  assert(!productTerms.empty() && "No product terms");

  // Compute arrival times for product terms
  for (const auto &cube : sop.cubes) {
    SmallVector<DelayType> literalArrivalTimes;
    for (unsigned i = 0; i < sop.numVars; ++i)
      if (cube.mask[i])
        literalArrivalTimes.push_back(inputArrivalTimes[i]);

    if (!literalArrivalTimes.empty())
      productArrivalTimes.push_back(simulateAndTree(literalArrivalTimes));
  }

  // Build OR of all product terms with arrival time awareness
  return buildOr(builder, loc, productTerms, productArrivalTimes);
}

//===----------------------------------------------------------------------===//
// SOP Balancing Pattern
//===----------------------------------------------------------------------===//

/// Compute the delay from each input to the output in a SOP structure.
/// Simulates the actual tree construction to accurately predict delays.
/// Takes input arrival times into account.
static DelayType computeSOPDelays(const SOPForm &sop, unsigned numInputs,
                                  ArrayRef<DelayType> inputArrivalTimes,
                                  SmallVectorImpl<DelayType> &delays) {
  delays.resize(numInputs, 0);

  // Compute arrival time for each product term
  SmallVector<DelayType> productArrivalTimes;
  productArrivalTimes.reserve(sop.cubes.size());

  for (const auto &cube : sop.cubes) {
    SmallVector<DelayType> cubeInputTimes;
    for (unsigned i = 0; i < numInputs; ++i)
      if (cube.mask[i])
        cubeInputTimes.push_back(inputArrivalTimes[i]);

    DelayType productTime = simulateAndTree(cubeInputTimes);
    productArrivalTimes.push_back(productTime);
  }

  // Compute output arrival time through OR tree
  DelayType outputArrivalTime = simulateAndTree(productArrivalTimes);

  // For each input, compute its delay contribution
  for (unsigned inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
    // Find the critical path through this input
    DelayType maxDelay = 0;

    for (const auto &cube : sop.cubes) {
      if (!cube.mask[inputIdx])
        continue;

      maxDelay =
          std::max(maxDelay, outputArrivalTime - inputArrivalTimes[inputIdx]);
    }

    delays[inputIdx] = maxDelay;
  }

  return outputArrivalTime;
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
    LLVM_DEBUG({
      llvm::dbgs() << "Matching SOP form:\n";
      sop.dump(llvm::dbgs());
    });

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
    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting SOP form:\n";
      sop.dump(llvm::dbgs());
    });

    // Get input arrival times for delay-optimized balancing
    SmallVector<DelayType, 6> arrivalTimes;
    auto r = cut.getInputArrivalTimes(enumerator, arrivalTimes);
    (void)r;
    assert(succeeded(r) && "Failed to get input arrival times");

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
