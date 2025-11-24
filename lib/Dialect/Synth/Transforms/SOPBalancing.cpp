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

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
static APInt createVarMask(unsigned numVars, unsigned var, bool positive) {
  uint32_t numBits = 1u << numVars;
  APInt mask(numBits, 0);

  for (uint32_t i = 0; i < numBits; ++i) {
    // Check if bit i has variable 'var' set to the desired value
    bool varValue = (i & (1u << var)) != 0;
    if (varValue == positive)
      mask.setBit(i);
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

/// Compute cofactor: f_xi (positive) or f_!xi (negative).
/// Follows mockturtle/kitty semantics: duplicates the selected bits to fill
/// the entire truth table.
/// For cofactor0 (var=0): takes bits where var=0 and duplicates them
/// For cofactor1 (var=1): takes bits where var=1 and duplicates them
static APInt computeCofactor(const APInt &f, unsigned numVars, unsigned var,
                             bool positive) {
  uint32_t numBits = 1u << numVars;
  APInt result(numBits, 0);

  uint32_t blockSize = 1u << var;
  uint32_t numBlocks = numBits / (blockSize * 2);

  // Duplicate the selected bits to fill the entire truth table
  for (uint32_t block = 0; block < numBlocks; ++block) {
    uint32_t offset = block * blockSize * 2;

    if (positive) {
      // Positive cofactor: take var=1 bits (upper half) and duplicate
      for (uint32_t i = 0; i < blockSize; ++i) {
        bool bit = f[offset + blockSize + i];
        if (bit) {
          result.setBit(offset + i);             // Copy to lower half
          result.setBit(offset + blockSize + i); // Copy to upper half
        }
      }
    } else {
      // Negative cofactor: take var=0 bits (lower half) and duplicate
      for (uint32_t i = 0; i < blockSize; ++i) {
        bool bit = f[offset + i];
        if (bit) {
          result.setBit(offset + i);             // Copy to lower half
          result.setBit(offset + blockSize + i); // Copy to upper half
        }
      }
    }
  }

  return result;
}

/// Check if a variable actually affects the function by comparing cofactors.
static bool hasVar(const APInt &f, unsigned numVars, unsigned var) {
  APInt f0 = computeCofactor(f, numVars, var, false);
  APInt f1 = computeCofactor(f, numVars, var, true);
  return f0 != f1;
}

/// Minato-Morreale ISOP algorithm (direct port from mockturtle/kitty).
/// This recursively computes an irredundant sum-of-products form.
///
/// Parameters:
///   tt: The ON-set (truth table to cover)
///   dc: The don't-care set (can be used to cover ON-set)
///   numVars: Total number of variables
///   varIndex: Current variable index (counts down from numVars to 0)
///   cubes: Output vector of cubes
///
/// Returns: The actual care set covered by the generated cubes
static APInt isopRec(const APInt &tt, const APInt &dc, unsigned numVars,
                     unsigned varIndex, SOPForm &result) {
  // Invariant: tt must be a subset of dc (all ON-set bits are in care set)
  assert((tt & ~dc).isZero() && "tt must be subset of dc");

  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: all don't-cares, add empty cube
  if ((~dc).isZero()) {
    result.cubes.emplace_back(numVars);
    return dc;
  }

  // Base case: no more variables to process
  if (varIndex == 0) {
    // If we still have minterms to cover, add an empty cube
    if (!tt.isZero()) {
      result.cubes.emplace_back(numVars);
    }
    return tt;
  }

  // Find the highest variable that actually appears in tt or dc
  int var = varIndex - 1;
  for (; var >= 0; --var) {
    if (hasVar(tt, numVars, var) || hasVar(dc, numVars, var))
      break;
  }

  // If no variable found, add empty cube if needed
  assert(var >= 0 && "No variable found in tt or dc");

  // Compute cofactors
  APInt tt0 = computeCofactor(tt, numVars, var, false);
  APInt tt1 = computeCofactor(tt, numVars, var, true);
  APInt dc0 = computeCofactor(dc, numVars, var, false);
  APInt dc1 = computeCofactor(dc, numVars, var, true);

  // Track cube indices for adding literals later
  size_t beg0 = result.cubes.size();
  APInt res0 = isopRec(tt0 & ~dc1, dc0, numVars, var, result);
  size_t end0 = result.cubes.size();

  APInt res1 = isopRec(tt1 & ~dc0, dc1, numVars, var, result);
  size_t end1 = result.cubes.size();

  APInt res2 =
      isopRec((tt0 & ~res0) | (tt1 & ~res1), dc0 & dc1, numVars, var, result);

  // Create masks for the variable
  APInt var0Mask = createVarMask(numVars, var, false); // Minterms where var=0
  APInt var1Mask = createVarMask(numVars, var, true);  // Minterms where var=1

  // Combine results: res0 is restricted to var=0, res1 to var=1
  res2 |= (res0 & var0Mask) | (res1 & var1Mask);

  // Add literals to cubes generated in the first recursion (var=0)
  for (size_t c = beg0; c < end0; ++c) {
    result.cubes[c].mask.setBit(var);
    result.cubes[c].inverted.setBit(var); // Negative literal
  }

  // Add literals to cubes generated in the second recursion (var=1)
  for (size_t c = end0; c < end1; ++c) {
    result.cubes[c].mask.setBit(var);
    // inverted bit remains 0 for positive literal
  }

  // Verify invariants
  assert((tt & ~res2).isZero() && "result must cover tt");
  assert((res2 & ~dc).isZero() && "result must be subset of dc");

  return res2;
}

/// Extract ISOP (Irredundant Sum-of-Products) from truth table.
/// Uses Minato-Morreale algorithm for efficient ISOP computation.
static SOPForm extractSOPFromTruthTable(const BinaryTruthTable &tt) {
  SOPForm sop(tt.numInputs);

  if (tt.numInputs == 0 || tt.table.isZero())
    return sop;

  // Call the mockturtle-style ISOP algorithm
  // dc = tt means all ON-set bits are also don't-cares (no OFF-set constraints)
  (void)isopRec(tt.table, tt.table, tt.numInputs, tt.numInputs, sop);

  // Verify the result is correct
  APInt result = sop.computeTruthTable();
  if (result != tt.table) {
    llvm::errs() << "ISOP does not match original truth table!\n";
    llvm::errs() << "Original: " << tt.table << "\n";
    llvm::errs() << "ISOP: " << result << "\n";
    sop.dump(llvm::errs());
    tt.dump(llvm::errs());
  }
  assert(result == tt.table && "ISOP does not match original truth table!");

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
      tt.dump(llvm::dbgs());
      llvm::dbgs() << "Matching SOP form:\n";
      sop.dump(llvm::dbgs());
      llvm::dbgs() << "Is irredundant: " << sop.isIrredundant() << "\n";
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
