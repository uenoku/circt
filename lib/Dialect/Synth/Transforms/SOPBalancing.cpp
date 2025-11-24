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
/// Uses bit manipulation for efficient computation.
static APInt computeCofactor(const APInt &f, unsigned numVars, unsigned var,
                             bool positive) {
  uint32_t numBits = 1u << numVars;
  APInt result(numBits, 0);

  // Use bit manipulation to compute cofactor efficiently.
  // The cofactor operation selects and compacts bits based on variable value.

  uint32_t blockSize = 1u << var; // Size of each block to process
  uint32_t numBlocks = numBits / (blockSize * 2); // Number of block pairs

  // Process each block pair
  for (uint32_t block = 0; block < numBlocks; ++block) {
    uint32_t srcOffset = block * blockSize * 2;
    uint32_t dstOffset = block * blockSize;

    if (positive) {
      // Positive cofactor: copy upper half of each block pair
      // Use extractBits for efficient bulk copy when blockSize is large
      if (blockSize >= 64) {
        APInt extracted = f.extractBits(blockSize, srcOffset + blockSize);
        result.insertBits(extracted, dstOffset);
      } else {
        for (uint32_t i = 0; i < blockSize; ++i)
          if (f[srcOffset + blockSize + i])
            result.setBit(dstOffset + i);
      }
    } else {
      // Negative cofactor: copy lower half of each block pair
      // Use extractBits for efficient bulk copy when blockSize is large
      if (blockSize >= 64) {
        APInt extracted = f.extractBits(blockSize, srcOffset);
        result.insertBits(extracted, dstOffset);
      } else {
        for (uint32_t i = 0; i < blockSize; ++i)
          if (f[srcOffset + i])
            result.setBit(dstOffset + i);
      }
    }
  }

  return result;
}

struct TruthTableWithDC {
  APInt tt;
  APInt dc;
};

static TruthTableWithDC computeCofactor(const TruthTableWithDC &f,
                                        unsigned numVars, unsigned var,
                                        bool positive) {
  TruthTableWithDC result;
  result.tt = computeCofactor(f.tt, numVars, var, positive);
  result.dc = computeCofactor(f.dc, numVars, var, positive);
  return result;
}

template <bool positive>
static APInt computeCofactor(const APInt &f, unsigned numVars, unsigned var) {
  uint32_t numBits = 1u << numVars;
  APInt result(numBits, 0);

  // Use bit manipulation to compute cofactor efficiently.
  // The cofactor operation can be viewed as selecting and compacting bits.
  // For positive cofactor f_x: select bits where var=1, compact to lower half
  // For negative cofactor f_!x: select bits where var=0, compact to lower half

  // Optimization: For var >= 6 (blockSize >= 64), we can use word-level
  // operations by directly accessing APInt's internal representation via
  // getRawData(). For smaller var, the overhead of bit-by-bit operations is
  // acceptable.

  uint32_t blockSize = 1u << var; // Size of each block to process
  uint32_t numBlocks = numBits / (blockSize * 2); // Number of block pairs

  // Process each block pair
  for (uint32_t block = 0; block < numBlocks; ++block) {
    uint32_t srcOffset = block * blockSize * 2;
    uint32_t dstOffset = block * blockSize;

    if constexpr (positive) {
      // Positive cofactor: copy upper half of each block pair
      // Use extractBits for efficient bulk copy when blockSize is large
      if (blockSize >= 64) {
        APInt extracted = f.extractBits(blockSize, srcOffset + blockSize);
        result.insertBits(extracted, dstOffset);
      } else {
        for (uint32_t i = 0; i < blockSize; ++i)
          if (f[srcOffset + blockSize + i])
            result.setBit(dstOffset + i);
      }
    } else {
      // Negative cofactor: copy lower half of each block pair
      // Use extractBits for efficient bulk copy when blockSize is large
      if (blockSize >= 64) {
        APInt extracted = f.extractBits(blockSize, srcOffset);
        result.insertBits(extracted, dstOffset);
      } else {
        for (uint32_t i = 0; i < blockSize; ++i)
          if (f[srcOffset + i])
            result.setBit(dstOffset + i);
      }
    }
  }

  return result;
}
/// The Recursive Minato-Morreale ISOP Algorithm.
/// func: The truth table as a bit vector (size must be power of 2)
/// currentVars: The number of variables currently in scope
/// cube: The current cube being built (accumulates literals from parent calls)
/// result: The SOPForm to append cubes to
static void minatoIsop(const APInt &func, unsigned currentVars,
                       const Cube &cube, SOPForm &result) {
  // Base case: If function is 0 (False), no cubes cover it
  if (func.isZero())
    return;

  // Base case: If function is all 1s (True), add the current cube
  if (func.isAllOnes()) {
    result.cubes.push_back(cube);
    return;
  }

  // Recursive step: split on the highest variable in current scope
  unsigned splitVarIdx = currentVars - 1;
  unsigned halfSize = func.getBitWidth() / 2;

  // Split truth table into two halves:
  // Lower half corresponds to splitVar = 0
  // Upper half corresponds to splitVar = 1
  APInt f0 = func.trunc(halfSize);
  APInt f1 = func.lshr(halfSize).trunc(halfSize);

  // Minato-Morreale decomposition into three disjoint parts:
  // 1. Shared part (f0 & f1) -> independent of splitVar
  APInt fShared = f0 & f1;

  // 2. Unique to 0 (f0 & ~f1) -> requires splitVar = 0
  APInt fUnique0 = f0 & ~f1;

  // 3. Unique to 1 (f1 & ~f0) -> requires splitVar = 1
  APInt fUnique1 = f1 & ~f0;

  // Recursion 1: Shared terms (do NOT add literal for splitVar)
  minatoIsop(fShared, currentVars - 1, cube, result);

  // Recursion 2: Unique 0 terms (add negative literal !splitVar)
  Cube cube0 = cube;
  cube0.mask.setBit(splitVarIdx);
  cube0.inverted.setBit(splitVarIdx);
  minatoIsop(fUnique0, currentVars - 1, cube0, result);

  // Recursion 3: Unique 1 terms (add positive literal splitVar)
  Cube cube1 = cube;
  cube1.mask.setBit(splitVarIdx);
  // inverted bit remains 0 for positive literal
  minatoIsop(fUnique1, currentVars - 1, cube1, result);
}

/// Minato-Morreale ISOP algorithm with don't-cares.
/// Recursively computes an irredundant sum-of-products form.
/// Returns the care set covered by the generated cubes.
static APInt isopRecursiveWithDC(const APInt &tt, const APInt &dc,
                                 unsigned numVars, unsigned varIndex,
                                 SOPForm &sop) {
  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: all don't-cares, add empty cube (constant 1)
  if (dc.isAllOnes()) {
    sop.cubes.emplace_back(numVars);
    return dc;
  }

  // Base case: ran out of variables
  if (varIndex >= numVars) {
    // If we still have minterms to cover, add them as cubes
    if (!tt.isZero())
      sop.cubes.emplace_back(numVars);
    return tt | dc;
  }

  // Compute cofactors for current variable
  auto negativeTT = computeCofactor<false>(tt, numVars, varIndex);
  auto negativeDC = computeCofactor<false>(dc, numVars, varIndex);
  auto positiveTT = computeCofactor<true>(tt, numVars, varIndex);
  auto positiveDC = computeCofactor<true>(dc, numVars, varIndex);

  // Track cube indices for adding literals later
  const auto beg0 = sop.cubes.size();

  // Recurse on negative cofactor (var = 0)
  // Cover minterms that are only in negative cofactor
  const auto res0 = isopRecursiveWithDC(negativeTT & ~positiveDC, negativeDC,
                                        numVars, varIndex + 1, sop);
  const auto end0 = sop.cubes.size();

  // Recurse on positive cofactor (var = 1)
  // Cover minterms that are only in positive cofactor
  const auto res1 = isopRecursiveWithDC(positiveTT & ~negativeDC, positiveDC,
                                        numVars, varIndex + 1, sop);
  const auto end1 = sop.cubes.size();

  // Recurse on common part (minterms in both cofactors)
  // Cover remaining minterms that weren't covered by res0 or res1
  auto res2 =
      isopRecursiveWithDC((negativeTT & ~res0) | (positiveTT & ~res1),
                          negativeDC & positiveDC, numVars, varIndex + 1, sop);

  // Expand results back to original variable space
  // res0 corresponds to var=0, res1 to var=1, res2 to both
  APInt var0Mask = createVarMask(numVars, varIndex, false);
  APInt var1Mask = createVarMask(numVars, varIndex, true);
  res2 |= (res0 & var0Mask) | (res1 & var1Mask);

  // Add literals to cubes generated in negative cofactor
  for (auto c = beg0; c < end0; ++c) {
    sop.cubes[c].mask.setBit(varIndex);
    sop.cubes[c].inverted.setBit(varIndex); // Negative literal
  }

  // Add literals to cubes generated in positive cofactor
  for (auto c = end0; c < end1; ++c) {
    sop.cubes[c].mask.setBit(varIndex);
    // inverted bit remains 0 for positive literal
  }

  return res2;
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

  // Use the cleaner minatoIsop implementation
  Cube emptyCube(tt.numInputs);
  minatoIsop(tt.table, tt.numInputs, emptyCube, sop);

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
