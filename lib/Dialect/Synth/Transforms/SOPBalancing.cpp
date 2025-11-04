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

/// Precomputed masks for variables in truth tables up to 6 variables (64 bits).
/// Masks[var][0] = mask where var=0 (negative literal)
/// Masks[var][1] = mask where var=1 (positive literal)
static constexpr uint64_t kVarMasks[6][2] = {
    {0x5555555555555555ULL, 0xAAAAAAAAAAAAAAAAULL}, // var 0: alternating bits
    {0x3333333333333333ULL, 0xCCCCCCCCCCCCCCCCULL}, // var 1: pairs of bits
    {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL}, // var 2: nibbles
    {0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL}, // var 3: bytes
    {0x0000FFFF0000FFFFULL, 0xFFFF0000FFFF0000ULL}, // var 4: half-words
    {0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL}, // var 5: words
};

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
static APInt createVarMask(unsigned numVars, unsigned var, bool positive) {
  uint32_t numBits = 1u << numVars;

  // Use precomputed table for small cases (up to 6 variables = 64 bits)
  if (numVars <= 6) {
    assert(var < 6);
    uint64_t maskValue = kVarMasks[var][positive ? 1 : 0];
    // Mask off bits beyond numBits
    if (numBits < 64)
      maskValue &= (1ULL << numBits) - 1;
    return APInt(numBits, maskValue);
  }

  // For larger cases, build mask by setting bits in blocks
  APInt mask(numBits, 0);
  uint32_t shift = 1u << var;

  for (uint32_t i = 0; i < numBits; i += 2 * shift) {
    if (positive) {
      // Set upper half of each block
      for (uint32_t j = 0; j < shift && (i + shift + j) < numBits; ++j)
        mask.setBit(i + shift + j);
    } else {
      // Set lower half of each block
      for (uint32_t j = 0; j < shift && (i + j) < numBits; ++j)
        mask.setBit(i + j);
    }
  }

  return mask;
}

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
  APInt computeTruthTable() const {
    APInt tt(1 << numVars, 0);
    for (const auto &cube : cubes) {
      APInt cubeTT = ~APInt(1 << numVars, 0);
      for (unsigned i = 0; i < numVars; ++i) {
        if (cube.mask[i]) {
          cubeTT &= createVarMask(numVars, i, !cube.inverted[i]);
        }
      }
      tt |= cubeTT;
    }
    return tt;
  }

  bool isIrredundant() {
    APInt tt = computeTruthTable();
    for (auto &cube : cubes) {
      auto temporary = cube;
      // Remove one literal from the cube
      for (unsigned i = 0; i < numVars; ++i) {
        if (temporary.mask[i]) {
          cube.mask.setBitVal(i, 0);
          cube.inverted.setBitVal(i, 0);
          if (tt == computeTruthTable())
            return false;
          cube = temporary;
        }
      }
    }

    return true;
  }
};

/// Compute cofactor of a Boolean function.
///
/// The cofactor of a function f with respect to variable x is the function
/// obtained by fixing x to a constant value:
///   - Positive cofactor f_x  (or f|x=1): f with variable x set to 1
///   - Negative cofactor f_!x (or f|x=0): f with variable x set to 0
///
/// Example: f(x,y,z) = xy + !xz
///   - f_x  = f(1,y,z) = y       (when x=1, xy becomes y, !xz becomes 0)
///   - f_!x = f(0,y,z) = z       (when x=0, xy becomes 0, !xz becomes z)
///
/// In truth table representation (where bit i represents minterm i):
///   - For a 3-variable function, bits are indexed as: [xyz]
///     Bit 0: 000, Bit 1: 001, Bit 2: 010, Bit 3: 011,
///     Bit 4: 100, Bit 5: 101, Bit 6: 110, Bit 7: 111
///   - Negative cofactor (x=0): extract bits where x=0 (bits 0,1,2,3)
///   - Positive cofactor (x=1): extract bits where x=1 (bits 4,5,6,7)
///
/// Returns pair of (negative cofactor, positive cofactor).
static std::pair<APInt, APInt>
computeCofactors(const APInt &f, unsigned numVars, unsigned var) {
  uint32_t numBits = 1u << numVars;
  uint32_t shift = 1u << var;

  // Create mask that selects bits for each cofactor
  // For each 2*shift block: lower shift bits go to cof0, upper shift bits to
  // cof1
  APInt mask(numBits, 0);
  APInt blockMask = APInt::getLowBitsSet(numBits, shift);

  // Build masks for both cofactors in one pass
  APInt mask0(numBits, 0); // Selects bits where var=0
  APInt mask1(numBits, 0); // Selects bits where var=1

  for (uint32_t i = 0; i < numBits; i += 2 * shift) {
    mask0 |= blockMask.shl(i);         // Lower half of each block
    mask1 |= blockMask.shl(i + shift); // Upper half of each block
  }

  // Extract bits for each cofactor
  APInt selected0 = f & mask0;
  APInt selected1 = f & mask1;

  // Duplicate to fill entire truth table
  APInt cof0 = selected0 | selected0.shl(shift);  // Copy lower to upper
  APInt cof1 = selected1 | selected1.lshr(shift); // Copy upper to lower

  return {cof0, cof1};
}

/// Check if a variable is in the support of the function.
/// A variable is in the support if it actually affects the function output.
/// This is determined by comparing the two cofactors - if they differ, the
/// variable is in the support.
static bool variableInSupport(const APInt &f, unsigned numVars, unsigned var) {
  auto [f0, f1] = computeCofactors(f, numVars, var);
  return f0 != f1;
}

/// Minato-Morreale ISOP algorithm.
///
/// Computes an Irredundant Sum-of-Products (ISOP) cover for a Boolean function.
/// An ISOP is a sum-of-products where:
///   1. No cube can be removed without changing the function
///   2. The cubes are pairwise disjoint (no minterm is covered by multiple
///   cubes)
///
/// Parameters:
///   tt: The ON-set (minterms that must be covered)
///   dc: The don't-care set (minterms that can optionally be covered)
///       Invariant: tt ⊆ dc (all ON-set minterms are in the care set)
///   numVars: Total number of variables in the function
///   varIndex: Current variable index (counts down from numVars to 0)
///   result: Output SOP form (cubes are accumulated here)
///
static APInt isopRec(const APInt &tt, const APInt &dc, unsigned numVars,
                     unsigned varIndex, SOPForm &result) {
  // Invariant: tt must be a subset of dc (all ON-set bits are in care set)
  assert((tt & ~dc).isZero() && "tt must be subset of dc");

  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: all don't-cares, add empty cube
  if (dc.isAllOnes()) {
    result.cubes.emplace_back(numVars);
    return dc;
  }

  assert(varIndex > 0 && "No more variables to process");

  // Find the highest variable that actually appears in tt or dc
  int var = varIndex - 1;
  for (; var >= 0; --var)
    if (variableInSupport(tt, numVars, var) ||
        variableInSupport(dc, numVars, var))
      break;

  // If no variable found, add empty cube if needed
  assert(var >= 0 && "No variable found in tt or dc");

  // Compute cofactors with respect to the splitting variable
  auto [negativeCofactor, positiveCofactor] =
      computeCofactors(tt, numVars, var);
  auto [negativeDC, positiveDC] = computeCofactors(dc, numVars, var);

  // Recurse on minterms unique to negative cofactor (will get !var literal)
  size_t negativeBegin = result.cubes.size();
  APInt negativeCover =
      isopRec(negativeCofactor & ~positiveDC, negativeDC, numVars, var, result);
  size_t negativeEnd = result.cubes.size();

  // Recurse on minterms unique to positive cofactor (will get var literal)
  APInt positiveCover =
      isopRec(positiveCofactor & ~negativeDC, positiveDC, numVars, var, result);
  size_t positiveEnd = result.cubes.size();

  // Recurse on shared minterms (will get no literal for this variable)
  APInt sharedCover = isopRec((negativeCofactor & ~negativeCover) |
                                  (positiveCofactor & ~positiveCover),
                              negativeDC & positiveDC, numVars, var, result);

  // Create masks for the variable to restrict covers to their domains
  APInt negativeMask =
      createVarMask(numVars, var, false); // Minterms where var=0
  APInt positiveMask =
      createVarMask(numVars, var, true); // Minterms where var=1

  // Combine results: restrict each cover to its domain
  APInt totalCover = sharedCover | (negativeCover & negativeMask) |
                     (positiveCover & positiveMask);

  // Add negative literal to cubes from first recursion
  APInt mask(numVars, 1 << var);
  for (size_t i = negativeBegin; i < negativeEnd; ++i) {
    result.cubes[i].mask |= mask;
    result.cubes[i].inverted |= mask;
  }

  // Add positive literal to cubes from second recursion
  for (size_t i = negativeEnd; i < positiveEnd; ++i) {
    result.cubes[i].mask |= mask;
    // inverted bit remains 0 for positive literal var
  }

  // Verify invariants
  assert((tt & ~totalCover).isZero() && "result must cover tt");
  assert((totalCover & ~dc).isZero() && "result must be subset of dc");

  return totalCover;
}

/// Extract ISOP (Irredundant Sum-of-Products) from truth table.
/// Uses Minato-Morreale algorithm for efficient ISOP computation.
static SOPForm extractSOPFromTruthTable(const BinaryTruthTable &tt) {
  SOPForm sop(tt.numInputs);

  if (tt.numInputs == 0 || tt.table.isZero())
    return sop;

  // Call the ISOP algorithm
  // dc = tt means all ON-set bits are also don't-cares (no OFF-set constraints)
  (void)isopRec(tt.table, tt.table, tt.numInputs, tt.numInputs, sop);

// Verify the result is correct
#ifdef DEBUG
  APInt result = sop.computeTruthTable();
  (void)result;
  assert(result == tt.table && "ISOP does not match original truth table!");
#endif

  return sop;
}

/// Build a balanced AND/OR tree from a list of values.
/// Builds a balanced tree based on arrival times to minimize delay.
/// Uses a priority queue to greedily pair nodes with earliest arrival times.
///
/// For AND: combines values with specified inversions
/// For OR: uses De Morgan's law: OR(a,b,...) = NOT(AND(NOT a, NOT b, ...))
template <bool isOr>
static Value buildBalancedTree(OpBuilder &builder, Location loc,
                               ArrayRef<Value> values, ArrayRef<bool> inverted,
                               ArrayRef<DelayType> arrivalTimes) {
  assert(values.size() == inverted.size() && "Size mismatch");
  assert(values.size() == arrivalTimes.size() && "Arrival times size mismatch");

  if (values.empty())
    return {};

  if (values.size() == 1)
    return aig::AndInverterOp::create(builder, loc, values[0], inverted[0]);

  // Build balanced tree based on arrival times
  SmallVector<ValueWithArrivalTime> nodes;
  size_t valueNumber = 0;
  for (unsigned i = 0; i < values.size(); ++i) {
    // For OR, invert all inputs (De Morgan's law)
    bool inv = isOr ? !inverted[i] : inverted[i];
    nodes.push_back(
        ValueWithArrivalTime(values[i], arrivalTimes[i], inv, valueNumber++));
  }

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
            // For OR, result stays inverted (De Morgan's law)
            return ValueWithArrivalTime(andResult, newTime, isOr,
                                        valueNumber++);
          });

  // Apply final inversion if needed
  if (result.isInverted())
    return aig::AndInverterOp::create(builder, loc, result.getValue(), true);
  return result.getValue();
}

/// Build an AND operation from a list of values.
static Value buildAnd(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                      ArrayRef<bool> inverted,
                      ArrayRef<DelayType> arrivalTimes) {
  return buildBalancedTree<false>(builder, loc, values, inverted, arrivalTimes);
}

/// Build an OR operation from a list of values.
static Value buildOr(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                     ArrayRef<DelayType> arrivalTimes) {
  SmallVector<bool> inverted(values.size(), false);
  return buildBalancedTree<true>(builder, loc, values, inverted, arrivalTimes);
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
  SOPBalancingPattern(MLIRContext *context) : CutRewritePattern(context) {}

  std::optional<MatchResult> match(const Cut &cut,
                                   CutEnumerator &enumerator) const override {
    // Match any non-trivial cut with a single output
    if (cut.isTrivialCut() || cut.getOutputSize() != 1)
      return std::nullopt;

    // Get the truth table for the cut
    const auto &tt = cut.getTruthTable();

    // Extract SOP form from truth table
    // TODO: mockturtle nicely parametrizes sop/bi-decomposition/aiker etc as
    // "resynthesis" engine. Consider taking similar approach if we want to do
    // more than SOP.
    SOPForm sop = extractSOPFromTruthTable(tt);
    LLVM_DEBUG({
      tt.dump(llvm::dbgs());
      llvm::dbgs() << "Matching SOP form:\n";
      sop.dump(llvm::dbgs());
      llvm::dbgs() << "Is irredundant: " << sop.isIrredundant() << "\n";
    });

    // If SOP is empty, don't match
    if (sop.cubes.empty())
      return std::nullopt;

    // Compute area (number of gates in the balanced structure)
    // This is a rough estimate: sum of cube sizes + OR gates
    unsigned totalGates = 0;
    for (const auto &cube : sop.cubes) {
      if (cube.size() > 1)
        totalGates += cube.size() - 1; // AND gates in this cube
    }
    if (sop.cubes.size() > 1)
      totalGates += sop.cubes.size() - 1; // OR gates

    // Compute delays from each input to the output.
    SmallVector<DelayType, 6> arrivalTimes;
    if (failed(cut.getInputArrivalTimes(enumerator, arrivalTimes)))
      return std::nullopt;

    // Compute delays using input arrival times and transfer ownership
    SmallVector<DelayType, 6> computedDelays;
    computeSOPDelays(sop, cut.getInputSize(), arrivalTimes, computedDelays);

    // Create and return match result with owned delays
    MatchResult result;
    result.area = static_cast<double>(totalGates);
    result.setOwnedDelays(std::move(computedDelays));
    return result;
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
