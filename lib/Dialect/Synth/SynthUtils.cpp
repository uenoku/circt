//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace circt::synth;
using llvm::APInt;

//===----------------------------------------------------------------------===//
// ISOP (Irredundant Sum-of-Products) Implementation
//===----------------------------------------------------------------------===//

/// Precomputed masks for variables in truth tables up to 6 variables (64 bits).
///
/// In a truth table, bit position i represents minterm i, where the binary
/// representation of i gives the variable values. For example, with 3 variables
/// (a,b,c), bit 5 (binary 101) represents minterm a=1, b=0, c=1.
///
/// These masks identify which minterms have a particular variable value:
/// - Masks[0][var] = minterms where var=0 (for negative literal !var)
/// - Masks[1][var] = minterms where var=1 (for positive literal var)
static constexpr uint64_t kVarMasks[2][6] = {
    {0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
     0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL,
     0x00000000FFFFFFFFULL}, // var=0 masks
    {0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
     0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL,
     0xFFFFFFFF00000000ULL}, // var=1 masks
};

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
template <bool positive>
static APInt createVarMask(unsigned numVars, unsigned var) {
  uint64_t numBits = 1u << numVars;

  // Use precomputed table for small cases (up to 6 variables = 64 bits)
  if (numVars <= 6) {
    assert(var < 6);
    uint64_t maskValue = kVarMasks[positive][var];
    // Mask off bits beyond numBits
    if (numBits < 64)
      maskValue &= (1ULL << numBits) - 1;
    return APInt(numBits, maskValue);
  }

  // For larger cases, build mask by setting bits in blocks
  APInt mask(numBits, 0);
  uint64_t shift = 1u << var;

  for (uint64_t i = 0; i < numBits; i += 2 * shift) {
    if (positive) {
      // Set upper half of each block
      for (uint64_t j = 0; j < shift && (i + shift + j) < numBits; ++j)
        mask.setBit(i + shift + j);
    } else {
      // Set lower half of each block
      for (uint64_t j = 0; j < shift && (i + j) < numBits; ++j)
        mask.setBit(i + j);
    }
  }

  return mask;
}

static APInt createVarMask(unsigned numVars, unsigned var, bool positive) {
  if (positive)
    return createVarMask<true>(numVars, var);
  return createVarMask<false>(numVars, var);
}

/// Compute cofactor of a Boolean function for a given variable.
///
/// A cofactor of a function f with respect to variable x is the function
/// obtained by fixing x to a constant value:
///   - Negative cofactor f|x=0 (or f_x'): f with variable x set to 0
///   - Positive cofactor f|x=1 (or f_x):  f with variable x set to 1
///
/// Cofactors are fundamental in Boolean function decomposition and the
/// Shannon expansion: f = x'*f|x=0 + x*f|x=1
///
/// In truth table representation, cofactors are computed by selecting the
/// subset of minterms where the variable has the specified value, then
/// replicating that pattern across the full truth table width.
///
/// Returns: (negative cofactor, positive cofactor)
static std::pair<APInt, APInt>
computeCofactors(const APInt &f, unsigned numVars, unsigned var) {
  uint64_t numBits = 1u << numVars;
  uint64_t shift = 1u << var;

  // Create mask that selects bits for each cofactor
  APInt blockMask = APInt::getLowBitsSet(numBits, shift);

  // Build masks for both cofactors in one pass
  APInt mask0(numBits, 0); // Selects bits where var=0
  APInt mask1(numBits, 0); // Selects bits where var=1

  for (uint64_t i = 0; i < numBits; i += 2 * shift) {
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
///
/// The support of a Boolean function is the set of variables that the function
/// depends on. A variable x is in the support if changing its value can change
/// the function's output. Formally, x is in the support if f|x=0 != f|x=1
/// (the positive and negative cofactors differ).
///
/// Example: f(a,b,c) = a*b has support {a,b} since c doesn't affect the output.
static bool variableInSupport(const APInt &f, unsigned numVars, unsigned var) {
  auto [f0, f1] = computeCofactors(f, numVars, var);
  return f0 != f1;
}

/// Minato-Morreale ISOP algorithm.
///
/// Computes an Irredundant Sum-of-Products (ISOP) cover for a Boolean function.
/// An ISOP is a sum-of-products representation where:
///   1. No cube can be removed without changing the function (irredundancy)
///   2. The cubes are pairwise disjoint (no minterm is covered by multiple
///   cubes)
///
/// The algorithm uses Shannon decomposition to recursively partition the
/// function based on a selected variable, computing covers for:
///   - Minterms unique to the negative cofactor (get !var literal)
///   - Minterms unique to the positive cofactor (get var literal)
///   - Minterms common to both cofactors (no literal for this variable)
///
/// Parameters:
///   onSet: The ON-set (minterms that must be covered)
///   dontCareSet: The don't-care set (minterms that can be included in the
///   cover)
///                Invariant: onSet ⊆ dontCareSet (all ON-set minterms are
///                don't-cares)
///   numVars: Total number of variables in the function
///   varIndex: Current variable index (counts down from numVars to 0)
///   result: Output SOP form (cubes are accumulated here)
///
/// Returns: The set of minterms covered by the computed ISOP.
static APInt isopImpl(const APInt &onSet, const APInt &dontCareSet,
                      unsigned numVars, unsigned varIndex, SOPForm &result) {
  // Invariant: onSet must be a subset of dontCareSet
  assert((onSet & ~dontCareSet).isZero() &&
         "onSet must be subset of dontCareSet");

  // Base case: nothing to cover
  if (onSet.isZero())
    return onSet;

  // Base case: all don't-cares, add empty cube
  if (dontCareSet.isAllOnes()) {
    result.cubes.emplace_back(numVars);
    return dontCareSet;
  }

  assert(varIndex > 0 && "No more variables to process");

  // Find a splitting variable and compute its cofactors
  int var = -1;
  APInt negativeCofactor, positiveCofactor, negativeDC, positiveDC;
  for (int v = varIndex - 1; v >= 0; --v) {
    std::tie(negativeCofactor, positiveCofactor) =
        computeCofactors(onSet, numVars, v);
    std::tie(negativeDC, positiveDC) =
        computeCofactors(dontCareSet, numVars, v);

    // Check if variable is in support of onSet or dontCareSet.
    if (negativeCofactor != positiveCofactor || negativeDC != positiveDC) {
      var = v;
      break;
    }
  }

  assert(var >= 0 && "No variable found in onSet or dontCareSet");

  // Recurse on minterms unique to negative cofactor
  size_t negativeBegin = result.cubes.size();
  APInt negativeCover = isopImpl(negativeCofactor & ~positiveDC, negativeDC,
                                 numVars, var, result);
  size_t negativeEnd = result.cubes.size();

  // Recurse on minterms unique to positive cofactor
  APInt positiveCover = isopImpl(positiveCofactor & ~negativeDC, positiveDC,
                                 numVars, var, result);
  size_t positiveEnd = result.cubes.size();

  // Recurse on shared minterms
  APInt sharedCover = isopImpl((negativeCofactor & ~negativeCover) |
                                   (positiveCofactor & ~positiveCover),
                               negativeDC & positiveDC, numVars, var, result);

  // Create masks for the variable to restrict covers to their domains
  APInt negativeMask =
      createVarMask<false>(numVars, var); // Minterms where var=0
  APInt positiveMask =
      createVarMask<true>(numVars, var); // Minterms where var=1

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
  for (size_t i = negativeEnd; i < positiveEnd; ++i)
    result.cubes[i].mask |= mask;

  assert((onSet & ~totalCover).isZero() && "result must cover onSet");
  assert((totalCover & ~dontCareSet).isZero() &&
         "result must be subset of dontCareSet");

  return totalCover;
}

void SOPForm::dump(llvm::raw_ostream &os) const {
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

/// Compute the truth table represented by the SOP form.
APInt SOPForm::computeTruthTable() const {
  APInt tt(1 << numVars, 0);
  for (const auto &cube : cubes) {
    APInt cubeTT = ~APInt(1 << numVars, 0);
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.mask[i])
        cubeTT &= createVarMask(numVars, i, !cube.inverted[i]);
    }
    tt |= cubeTT;
  }
  return tt;
}

// Check if the SOP form is irredundant (no cube can be removed).
// This is done by attempting to remove each literal from each cube
// and checking if the overall truth table remains the same.
// This is a brute-force check that verifies the irredundancy property and
// shouldn't be used as part of the ISOP extraction itself.
bool SOPForm::isIrredundant() {
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

SOPForm circt::synth::extractISOPFromTruthTable(const APInt &truthTable) {
  auto numVars = llvm::Log2_64_Ceil(truthTable.getBitWidth());
  assert((1u << numVars) == truthTable.getBitWidth() &&
         "Truth table size must be a power of two");
  SOPForm sop(numVars);

  if (numVars == 0 || truthTable.isZero())
    return sop;

  // Call the ISOP algorithm
  // dontCareSet = onSet means all ON-set bits are don't-cares (no OFF-set
  // constraints)
  auto result = isopImpl(truthTable, truthTable, numVars, numVars, sop);

#ifdef DEBUG
  // Verify the result is correct
  APInt tt = sop.computeTruthTable();
  (void)tt;
  assert(result == tt && "ISOP does not match original truth table!");
  assert(result == truthTable && "ISOP does not match original truth table!");
#endif

  return sop;
}
