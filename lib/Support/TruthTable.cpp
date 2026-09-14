//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements truth table utilities.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TruthTable.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <cassert>

using namespace circt;
using llvm::APInt;

//===----------------------------------------------------------------------===//
// BinaryTruthTable
//===----------------------------------------------------------------------===//

namespace {

static llvm::APInt
expandTruthTableToInputSpaceSmall(const llvm::APInt &tt,
                                  ArrayRef<unsigned> mapping,
                                  unsigned numExpandedInputs) {
  assert(numExpandedInputs <= 6 && "small fast path requires <= 6 inputs");

  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;
  // `kVarMasks[i]` marks rows where expanded input `i` is 1. For example,
  // `kVarMasks[0] = 0b1010...` and `kVarMasks[1] = 0b1100...`, matching the
  // standard packed truth-table row order where input 0 is the LSB.
  static constexpr uint64_t kVarMasks[6] = {
      0xAAAAAAAAAAAAAAAAULL, // input 0: 1010...
      0xCCCCCCCCCCCCCCCCULL, // input 1: 1100...
      0xF0F0F0F0F0F0F0F0ULL, // input 2: 11110000...
      0xFF00FF00FF00FF00ULL, // input 3: 8 zeros, 8 ones
      0xFFFF0000FFFF0000ULL, // input 4: 16 zeros, 16 ones
      0xFFFFFFFF00000000ULL  // input 5: 32 zeros, 32 ones
  };

  uint64_t sizeMask = llvm::maskTrailingOnes<uint64_t>(expandedSize);
  unsigned origSize = 1U << numOrigInputs;
  uint64_t origMask = llvm::maskTrailingOnes<uint64_t>(origSize);
  uint64_t origTT = tt.getZExtValue() & origMask;

  // `constrainMasks[i][b]` is the set of expanded rows where original input
  // `i`, after remapping into the expanded space, observes bit value `b`.
  std::array<std::array<uint64_t, 2>, 6> constrainMasks{};
  for (unsigned i = 0; i < numOrigInputs; ++i) {
    uint64_t posMask = kVarMasks[mapping[i]] & sizeMask;
    constrainMasks[i][0] = (~posMask) & sizeMask;
    constrainMasks[i][1] = posMask;
  }

  uint64_t activePatterns = origTT;
  bool useComplementResult = false;
  uint64_t result = 0;
  // If the original truth table has more 1s than 0s, it's more efficient to
  // iterate over the 0 patterns.
  if (static_cast<unsigned>(llvm::popcount(origTT)) > (origSize / 2)) {
    activePatterns = (~origTT) & origMask;
    useComplementResult = true;
    result = sizeMask;
  }

  while (activePatterns) {
    unsigned origIdx = llvm::countr_zero(activePatterns);
    activePatterns &= activePatterns - 1;

    uint64_t pattern = sizeMask;
    for (unsigned i = 0; i < numOrigInputs; ++i)
      pattern &= constrainMasks[i][(origIdx >> i) & 1U];

    if (useComplementResult)
      result &= ~pattern;
    else
      result |= pattern;
  }

  return llvm::APInt(expandedSize, result);
}

static llvm::APInt
expandTruthTableToInputSpaceGeneric(const llvm::APInt &tt,
                                    ArrayRef<unsigned> mapping,
                                    unsigned numExpandedInputs) {

  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;
  unsigned numOrigBits = tt.getBitWidth();

  // Pre-compute masks for each original input
  SmallVector<std::array<llvm::APInt, 2>, 16> masks;
  masks.reserve(numOrigInputs);
  for (unsigned i = 0; i < numOrigInputs; ++i) {
    unsigned mappedInput = mapping[i];
    masks.push_back({
        createVarMask(numExpandedInputs, mappedInput, false),
        createVarMask(numExpandedInputs, mappedInput, true),
    });
  }

  // Check sparsity
  unsigned numOnes = tt.popcount();
  bool enumerateSetBits = numOnes <= numOrigBits / 2;

  llvm::APInt result = enumerateSetBits ? llvm::APInt::getZero(expandedSize)
                                        : llvm::APInt::getAllOnes(expandedSize);

  for (unsigned origIdx = 0, e = numOrigBits; origIdx < e; ++origIdx) {
    if (tt[origIdx] != enumerateSetBits)
      continue;

    llvm::APInt pattern = llvm::APInt::getAllOnes(expandedSize);
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      bool bit = (origIdx >> i) & 1U;
      pattern &= masks[i][bit];
    }

    if (enumerateSetBits)
      result |= pattern;
    else
      result &= ~pattern;
  }
  return result;
}

} // namespace

llvm::APInt BinaryTruthTable::getOutput(const llvm::APInt &input) const {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
}

void BinaryTruthTable::setOutput(const llvm::APInt &input,
                                 const llvm::APInt &output) {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  assert(output.getBitWidth() == numOutputs && "Output width mismatch");
  assert(input.getBitWidth() < 32 && "Input width too large");
  unsigned offset = input.getZExtValue() * numOutputs;
  for (unsigned i = 0; i < numOutputs; ++i)
    table.setBitVal(offset + i, output[i]);
}

namespace {
/// Precomputed masks for variables in truth tables up to 6 variables (64 bits).
///
/// In a truth table, bit position i represents minterm i, where the binary
/// representation of i gives the variable values. For example, with 3 variables
/// (a,b,c), bit 5 (binary 101) represents minterm a=1, b=0, c=1.
///
/// These masks identify which minterms have a particular variable value:
/// - Masks[0][var] = minterms where var=0 (for negative literal !var)
/// - Masks[1][var] = minterms where var=1 (for positive literal var)
// NOTE: also used by createVarMaskImpl below; kept here so the word-level
// NPN transforms above can use the closed forms instead of rebuilding
// equivalent masks with bit loops per call.
static constexpr uint64_t kVarMasks[2][6] = {
    {0x5555555555555555ULL, 0x3333333333333333ULL, 0x0F0F0F0F0F0F0F0FULL,
     0x00FF00FF00FF00FFULL, 0x0000FFFF0000FFFFULL,
     0x00000000FFFFFFFFULL}, // var=0 masks
    {0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
     0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL,
     0xFFFFFFFF00000000ULL}, // var=1 masks
};

// All-ones mask for a single-output table with n <= 6 inputs (2^n bits).
uint64_t tableAllOnes(unsigned numInputs) {
  if (numInputs >= 6)
    return ~0ULL;
  return (1ULL << (1u << numInputs)) - 1;
}

// Swap two adjacent variables (bit positions pos and pos+1) in a packed
// single-output table. The middle minterms (var_pos=1,var_pos+1=0 and vice
// versa) exchange places; the masks are the closed forms of the previously
// loop-built sets: middleLow rows have var_pos=1,var_pos+1=0 and middleHigh
// rows the converse (pos <= 4 for n <= 6, so pos + 1 stays in bounds).
uint64_t swapAdjacentVariables(uint64_t table, unsigned numInputs,
                               unsigned pos) {
  unsigned s = 1u << pos;
  uint64_t widthMask = tableAllOnes(numInputs);
  uint64_t middleLow = kVarMasks[1][pos] & ~kVarMasks[1][pos + 1] & widthMask;
  uint64_t middleHigh = ~kVarMasks[1][pos] & kVarMasks[1][pos + 1] & widthMask;
  uint64_t rest = table & ~(middleLow | middleHigh);
  return rest | ((table & middleLow) << s) | ((table & middleHigh) >> s);
}
} // namespace

BinaryTruthTable
BinaryTruthTable::applyPermutation(ArrayRef<unsigned> permutation) const {
  assert(permutation.size() == numInputs && "Permutation size mismatch");
  if (numOutputs == 1 && numInputs <= 6) {
    // Word-level: decompose into adjacent variable swaps.
    uint64_t t = table.getZExtValue();
    SmallVector<unsigned> cur(numInputs);
    for (unsigned i = 0; i < numInputs; ++i)
      cur[i] = i;
    for (unsigned i = 0; i < numInputs; ++i) {
      unsigned k = i;
      while (cur[k] != permutation[i])
        ++k;
      while (k > i) {
        t = swapAdjacentVariables(t, numInputs, k - 1);
        std::swap(cur[k], cur[k - 1]);
        --k;
      }
    }
    return BinaryTruthTable(numInputs, numOutputs,
                            llvm::APInt(1u << numInputs, t));
  }
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt permutedInput = input;

    // Apply permutation
    for (unsigned j = 0; j < numInputs; ++j) {
      if (permutation[j] != j)
        permutedInput.setBitVal(j, input[permutation[j]]);
    }

    result.setOutput(permutedInput, getOutput(input));
  }

  return result;
}

BinaryTruthTable BinaryTruthTable::applyInputNegation(unsigned mask) const {
  if (numOutputs == 1 && numInputs <= 6) {
    // Word-level: negating variable v swaps adjacent blocks of size 2^v.
    // The var=1 rows are exactly the precomputed var mask (closed form of
    // the previously loop-built set).
    uint64_t t = table.getZExtValue();
    for (unsigned v = 0; v < numInputs; ++v) {
      if (!(mask & (1u << v)))
        continue;
      unsigned s = 1u << v;
      uint64_t hi = kVarMasks[1][v] & tableAllOnes(numInputs);
      t = ((t & hi) >> s) | ((t & ~hi) << s);
      t &= tableAllOnes(numInputs);
    }
    return BinaryTruthTable(numInputs, numOutputs,
                            llvm::APInt(1u << numInputs, t));
  }
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);

    // Apply negation using bitwise XOR
    llvm::APInt negatedInput = input ^ llvm::APInt(numInputs, mask);

    result.setOutput(negatedInput, getOutput(input));
  }

  return result;
}

BinaryTruthTable
BinaryTruthTable::applyOutputNegation(unsigned negation) const {
  if (numOutputs == 1 && numInputs <= 6) {
    uint64_t t = table.getZExtValue();
    if (negation & 1)
      t ^= tableAllOnes(numInputs);
    return BinaryTruthTable(numInputs, numOutputs,
                            llvm::APInt(1u << numInputs, t));
  }
  BinaryTruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt output = getOutput(input);

    // Apply negation using bitwise XOR
    llvm::APInt negatedOutput = output ^ llvm::APInt(numOutputs, negation);

    result.setOutput(input, negatedOutput);
  }

  return result;
}

bool BinaryTruthTable::isLexicographicallySmaller(
    const BinaryTruthTable &other) const {
  assert(numInputs == other.numInputs && numOutputs == other.numOutputs);
  return table.ult(other.table);
}

NCanonicalForm circt::computeNCanonicalForm(const BinaryTruthTable &tt) {
  if (tt.numOutputs != 1)
    return {tt, 0};
  NCanonicalForm best{tt, 0};
  for (uint32_t mask = 1; mask < (1u << tt.numInputs); ++mask) {
    BinaryTruthTable candidate = tt.applyInputNegation(mask);
    if (candidate.isLexicographicallySmaller(best.table))
      best = {std::move(candidate), mask};
  }
  return best;
}

bool BinaryTruthTable::operator==(const BinaryTruthTable &other) const {
  return numInputs == other.numInputs && numOutputs == other.numOutputs &&
         table == other.table;
}

void BinaryTruthTable::dump(llvm::raw_ostream &os) const {
  os << "TruthTable(" << numInputs << " inputs, " << numOutputs << " outputs, "
     << "value=";
  os << table << ")\n";

  // Print header
  for (unsigned i = 0; i < numInputs; ++i) {
    os << "i" << i << " ";
  }
  for (unsigned i = 0; i < numOutputs; ++i) {
    os << "o" << i << " ";
  }
  os << "\n";

  // Print separator
  for (unsigned i = 0; i < numInputs + numOutputs; ++i) {
    os << "---";
  }
  os << "\n";

  // Print truth table rows
  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    // Print input values
    for (unsigned j = 0; j < numInputs; ++j) {
      os << ((i >> j) & 1) << "  ";
    }

    // Print output values
    llvm::APInt input(numInputs, i);
    llvm::APInt output = getOutput(input);
    os << "|";
    for (unsigned j = 0; j < numOutputs; ++j) {
      os << output[j] << "  ";
    }
    os << "\n";
  }
}

llvm::APInt
circt::detail::expandTruthTableToInputSpace(const llvm::APInt &tt,
                                            ArrayRef<unsigned> mapping,
                                            unsigned numExpandedInputs) {
  unsigned numOrigInputs = mapping.size();
  unsigned expandedSize = 1U << numExpandedInputs;

  if (numOrigInputs == numExpandedInputs) {
    bool isIdentity = true;
    for (unsigned i = 0; i < numOrigInputs && isIdentity; ++i)
      isIdentity = mapping[i] == i;
    if (isIdentity)
      return tt.zext(expandedSize);
  }

  if (tt.isZero())
    return llvm::APInt::getZero(expandedSize);
  if (tt.isAllOnes())
    return llvm::APInt::getAllOnes(expandedSize);

  if (numExpandedInputs <= 6)
    return expandTruthTableToInputSpaceSmall(tt, mapping, numExpandedInputs);
  return expandTruthTableToInputSpaceGeneric(tt, mapping, numExpandedInputs);
}

/// Create a mask for a variable in the truth table.
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
template <bool positive>
static APInt createVarMaskImpl(unsigned numVars, unsigned varIndex) {
  assert(numVars <= 64 && "Number of variables too large");
  assert(varIndex < numVars && "Variable index out of bounds");
  uint64_t numBits = 1u << numVars;

  // Use precomputed table for small cases (up to 6 variables = 64 bits)
  if (numVars <= 6) {
    assert(varIndex < 6);
    uint64_t maskValue = kVarMasks[positive][varIndex];
    // Mask off bits beyond numBits
    if (numBits < 64)
      maskValue &= (1ULL << numBits) - 1;
    return APInt(numBits, maskValue);
  }

  // For larger cases, use getSplat to create repeating pattern
  // Pattern width is 2^(var+1) bits
  uint64_t patternWidth = 1u << (varIndex + 1);
  APInt pattern(patternWidth, 0);

  // Fill the appropriate half of the pattern
  uint64_t shift = 1u << varIndex;
  if constexpr (positive) {
    // Set upper half: bits [shift, patternWidth)
    pattern.setBits(shift, patternWidth);
  } else {
    // Set lower half: bits [0, shift)
    pattern.setBits(0, shift);
  }

  return APInt::getSplat(numBits, pattern);
}

APInt circt::createVarMask(unsigned numVars, unsigned varIndex, bool positive) {
  if (positive)
    return createVarMaskImpl<true>(numVars, varIndex);
  return createVarMaskImpl<false>(numVars, varIndex);
}

/// Compute cofactor of a Boolean function for a given variable.
///
/// A cofactor of a function f with respect to variable x is the function
/// obtained by fixing x to a constant value:
///   - Negative cofactor f|x=0 : f with variable x set to 0
///   - Positive cofactor f|x=1 : f with variable x set to 1
///
/// Cofactors are fundamental in Boolean function decomposition and the
/// Shannon expansion: f = x'*f|x=0 + x*f|x=1
///
/// In truth table representation, cofactors are computed by selecting the
/// subset of minterms where the variable has the specified value, then
/// replicating that pattern across the full truth table width.
///
/// Returns: (negative cofactor, positive cofactor)
std::pair<APInt, APInt>
circt::computeCofactors(const APInt &f, unsigned numVars, unsigned var) {
  assert(numVars <= 64 && "Number of variables too large");
  assert(var < numVars && "Variable index out of bounds");
  assert(f.getBitWidth() == (1u << numVars) && "Truth table size mismatch");
  uint64_t numBits = 1u << numVars;
  uint64_t shift = 1u << var;

  // Build masks using getSplat to replicate bit patterns
  // For var at position k, we need blocks of size 2^k where:
  // - mask0 selects lower 2^k bits of each 2^(k+1)-bit block (var=0)
  // - mask1 selects upper 2^k bits of each 2^(k+1)-bit block (var=1)
  APInt blockPattern = APInt::getLowBitsSet(2 * shift, shift);
  APInt mask0 = APInt::getSplat(numBits, blockPattern);
  APInt mask1 = mask0.shl(shift);

  // Extract bits for each cofactor
  APInt selected0 = f & mask0;
  APInt selected1 = f & mask1;

  // Replicate the selected bits across the full truth table
  APInt cof0 = selected0 | selected0.shl(shift);
  APInt cof1 = selected1 | selected1.lshr(shift);

  return {cof0, cof1};
}

//===----------------------------------------------------------------------===//
// SOPForm
//===----------------------------------------------------------------------===//

APInt SOPForm::computeTruthTable() const {
  APInt tt(1 << numVars, 0);
  for (const auto &cube : cubes) {
    APInt cubeTT = ~APInt(1 << numVars, 0);
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.hasLiteral(i))
        cubeTT &= createVarMask(numVars, i, !cube.isLiteralInverted(i));
    }
    tt |= cubeTT;
  }
  return tt;
}

#ifndef NDEBUG
void SOPForm::dump(llvm::raw_ostream &os) const {
  os << "SOPForm: " << numVars << " vars, " << cubes.size() << " cubes\n";
  for (const auto &cube : cubes) {
    os << "  (";
    for (unsigned i = 0; i < numVars; ++i) {
      if (cube.mask & (1ULL << i)) {
        os << ((cube.inverted & (1ULL << i)) ? "!" : "");
        os << "x" << i << " ";
      }
    }
    os << ")\n";
  }
}
#endif
//===----------------------------------------------------------------------===//
// ISOP Extraction
//===----------------------------------------------------------------------===//

namespace {

/// Minato-Morreale ISOP algorithm.
///
/// Computes an Irredundant Sum-of-Products (ISOP) cover for a Boolean function.
///
/// References:
/// - Minato, "Fast Generation of Irredundant Sum-of-Products Forms from Binary
///   Decision Diagrams", SASIMI 1992.
/// - Morreale, "Recursive Operators for Prime Implicant and Irredundant Normal
///   Form Determination", IEEE TC 1970.
///
/// Implementation inspired by lsils/kitty library:
/// https://github.com/lsils/kitty/blob/master/include/kitty/isop.hpp
/// distributed under MIT License.
///
/// The algorithm recursively decomposes the function using Shannon expansion:
///   f = !x * f|x=0 + x * f|x=1
///
/// For ISOP, we partition minterms into three groups:
/// - Minterms only in negative cofactor: must use !x literal
/// - Minterms only in positive cofactor: must use x literal
/// - Minterms in both cofactors: can omit x from the cube
///
/// Parameters:
///   tt: The ON-set (minterms that must be covered)
///   dc: The don't-care set (minterms that can optionally be covered)
///       Invariant: tt is a subset of dc (ON-set is subset of care set)
///   numVars: Total number of variables in the function
///   varIndex: Current variable index to start search from (counts down)
///   result: Output SOP form (cubes are accumulated here)
///
/// Returns: The actual cover computed (subset of dc that covers tt)
///
/// The maximum recursion depth is equal to the number of variables (one level
/// per variable). For typical use cases with TruthTable (up to 6-8 variables),
/// this is not a concern. Since truth tables require 2^numVars bits, the
/// recursion depth is not a limiting factor.
APInt isopImpl(const APInt &tt, const APInt &dc, unsigned numVars,
               unsigned varIndex, SOPForm &result) {
  assert((tt & ~dc).isZero() && "tt must be subset of dc");

  // Base case: nothing to cover
  if (tt.isZero())
    return tt;

  // Base case: function is tautology (all don't-cares)
  if (dc.isAllOnes()) {
    result.cubes.emplace_back();
    return dc;
  }

  // Find highest variable in support (top-down from varIndex).
  // NOTE: It is well known that the order of variable selection largely
  // affects the size of the resulting ISOP. There are numerous studies on
  // implementing heuristics for variable selection that could improve results,
  // albeit at the cost of runtime. We may consider implementing such
  // heuristics in the future.
  int var = -1;
  APInt negCof, posCof, negDC, posDC;
  for (int i = varIndex - 1; i >= 0; --i) {
    std::tie(negCof, posCof) = computeCofactors(tt, numVars, i);
    std::tie(negDC, posDC) = computeCofactors(dc, numVars, i);
    if (negCof != posCof || negDC != posDC) {
      var = i;
      break;
    }
  }
  assert(var >= 0 && "No variable found in support");

  // Recurse on minterms unique to negative cofactor (will get !var literal)
  size_t negBegin = result.cubes.size();
  APInt negCover = isopImpl(negCof & ~posDC, negDC, numVars, var, result);
  size_t negEnd = result.cubes.size();

  // Recurse on minterms unique to positive cofactor (will get var literal)
  APInt posCover = isopImpl(posCof & ~negDC, posDC, numVars, var, result);
  size_t posEnd = result.cubes.size();

  // Recurse on shared minterms (no literal for this variable)
  APInt remaining = (negCof & ~negCover) | (posCof & ~posCover);
  APInt sharedCover = isopImpl(remaining, negDC & posDC, numVars, var, result);

  // Compute total cover by restricting each sub-cover to its domain
  APInt negMask = createVarMaskImpl<false>(numVars, var);
  APInt totalCover = sharedCover | (negCover & negMask) | (posCover & ~negMask);

  // Add literals to cubes from recursions
  for (size_t i = negBegin; i < negEnd; ++i)
    result.cubes[i].setLiteral(var, true);

  for (size_t i = negEnd; i < posEnd; ++i)
    result.cubes[i].setLiteral(var, false);

  return totalCover;
}

} // namespace

SOPForm circt::extractISOP(const APInt &truthTable, unsigned numVars) {
  assert((1u << numVars) == truthTable.getBitWidth() &&
         "Truth table size must match 2^numVars");
  SOPForm sop(numVars);

  if (numVars == 0 || truthTable.isZero())
    return sop;

  (void)isopImpl(truthTable, truthTable, numVars, numVars, sop);

  return sop;
}
