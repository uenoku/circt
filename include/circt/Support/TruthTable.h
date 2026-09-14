//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines truth table utilities.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_TRUTHTABLE_H
#define CIRCT_SUPPORT_TRUTHTABLE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace circt {

/// Represents a boolean function as a truth table.
///
/// A truth table stores the output values for all possible input combinations
/// of a boolean function. For a function with n inputs and m outputs, the
/// truth table contains 2^n entries, each with m output bits.
///
/// Example: For a 2-input AND gate:
/// - Input 00 -> Output 0
/// - Input 01 -> Output 0
/// - Input 10 -> Output 0
/// - Input 11 -> Output 1
/// This would be stored as the bit pattern 0001 in the truth table.
///
/// TODO: Rename this to MultiOutputTruthTable since for single-output functions
///       we can just use llvm::APInt directly.
struct BinaryTruthTable {
  unsigned numInputs;  ///< Number of inputs for this boolean function
  unsigned numOutputs; ///< Number of outputs for this boolean function
  llvm::APInt table;   ///< Truth table data as a packed bit vector

  /// Default constructor creates an empty truth table.
  BinaryTruthTable() : numInputs(0), numOutputs(0), table(1, 0) {}

  /// Constructor for a truth table with given dimensions and evaluation data.
  BinaryTruthTable(unsigned numInputs, unsigned numOutputs,
                   const llvm::APInt &table)
      : numInputs(numInputs), numOutputs(numOutputs), table(table) {
    assert(table.getBitWidth() == (1u << numInputs) * numOutputs &&
           "Truth table size mismatch");
  }

  /// Constructor for a truth table with given dimensions, initialized to zero.
  BinaryTruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1u << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination.
  llvm::APInt getOutput(const llvm::APInt &input) const;

  /// Set the output value for a given input combination.
  void setOutput(const llvm::APInt &input, const llvm::APInt &output);

  /// Apply input permutation to create a new truth table.
  /// This reorders the input variables according to the given permutation.
  BinaryTruthTable applyPermutation(ArrayRef<unsigned> permutation) const;

  /// Apply input negation to create a new truth table.
  /// This negates selected input variables based on the mask.
  BinaryTruthTable applyInputNegation(unsigned mask) const;

  /// Apply output negation to create a new truth table.
  /// This negates selected output variables based on the mask.
  BinaryTruthTable applyOutputNegation(unsigned negation) const;

  /// Check if this truth table is lexicographically smaller than another.
  /// Used for canonical ordering of truth tables.
  bool isLexicographicallySmaller(const BinaryTruthTable &other) const;

  /// Equality comparison for truth tables.
  bool operator==(const BinaryTruthTable &other) const;

  /// Debug dump method for truth tables.
  void dump(llvm::raw_ostream &os = llvm::errs()) const;
};

/// N-canonical form of a boolean function: the lexicographically smallest
/// truth table over all input negation masks (no permutation search), plus
/// the achieving mask (first/smallest on ties).
///
/// Used for mockturtle-style matching: patterns pre-enumerate all pin
/// permutations at load time, so per-cut matching needs only this cheap
/// O(2^n) canonicalization plus a hash lookup, for both output polarities.
struct NCanonicalForm {
  BinaryTruthTable table;
  unsigned inputNegation = 0;
};

/// Compute the N-canonical form of a single-output truth table.
/// Non-single-output tables are returned unchanged with a zero mask.
NCanonicalForm computeNCanonicalForm(const BinaryTruthTable &tt);

/// Enumerate all pin permutations of a single-output truth table for
/// mockturtle-style pattern matching. For each permutation (lexicographic
/// order from the identity), calls fn with the N-canonical key table, the
/// pin map (pattern pin p is driven by cut input pinMap[p], i.e. the
/// inverse permutation), and the input-negation witness. Shared by
/// load-time pattern indexing and offline witness-table generation so the
/// two can never diverge.
void enumeratePatternEntries(
    const BinaryTruthTable &table,
    llvm::function_ref<void(const BinaryTruthTable &key,
                            ArrayRef<unsigned> pinMap, unsigned inNeg)>
        fn);

//===----------------------------------------------------------------------===//
// Truth Table Utilities
//===----------------------------------------------------------------------===//

/// Create a mask for a variable in the truth table.
///
/// This function generates a bitmask that identifies which minterms have a
/// particular value for a given variable in a truth table representation.
///
/// For positive=true: mask has 1s where var=1 in the truth table encoding
/// For positive=false: mask has 1s where var=0 in the truth table encoding
///
/// For example, with 3 variables (a,b,c) where c is LSB:
/// - createVarMask(3, 0, true) creates mask for c=1: 0b10101010
/// - createVarMask(3, 1, true) creates mask for b=1: 0b11001100
/// - createVarMask(3, 2, true) creates mask for a=1: 0b11110000
///
/// Parameters:
///   numVars: Number of variables in the truth table
///   var: Variable index (0 = LSB)
///   positive: true for var=1 mask, false for var=0 mask
///
/// Returns: APInt mask with numBits = 2^numVars
llvm::APInt createVarMask(unsigned numVars, unsigned varIndex, bool positive);

namespace detail {
/// Expand a truth table to a larger input space using the given input mapping.
/// `inputMapping[i]` gives the destination input position for original input
/// `i`.
llvm::APInt expandTruthTableToInputSpace(const llvm::APInt &tt,
                                         ArrayRef<unsigned> inputMapping,
                                         unsigned numExpandedInputs);
} // namespace detail

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
/// This function returns a pair of cofactors (negative, positive) for the
/// specified variable index.
std::pair<llvm::APInt, llvm::APInt>
computeCofactors(const llvm::APInt &f, unsigned numVars, unsigned varIndex);

//===----------------------------------------------------------------------===//
// ISOP (Irredundant Sum-of-Products) Utilities
//===----------------------------------------------------------------------===//

/// Represents a product term (cube) in a sum-of-products expression.
/// Each cube is a conjunction of literals (variables or their negations).
/// Assumes number of variables is less than 64.
struct Cube {
  /// Bitmask indicating which variables appear in this cube
  uint64_t mask = 0;
  /// Bitmask indicating which variables are negated
  uint64_t inverted = 0;

  Cube() = default;

  /// Get number of literals in this cube
  unsigned size() const { return llvm::popcount(mask); }

  /// Has literal for variable at index varIndex
  inline bool hasLiteral(unsigned varIndex) const {
    return mask & (1ULL << varIndex);
  }

  inline void setLiteral(unsigned varIndex, bool isInverted) {
    uint64_t varMask = 1ULL << varIndex;
    mask |= varMask;
    if (isInverted)
      inverted |= varMask;
  }
  inline void removeLiteral(unsigned varIndex) {
    uint64_t varMask = 1ULL << varIndex;
    mask &= ~varMask;
    inverted &= ~varMask;
  }

  inline bool isLiteralInverted(unsigned varIndex) const {
    return inverted & (1ULL << varIndex);
  }
};

/// Represents a sum-of-products expression.
struct SOPForm {
  llvm::SmallVector<Cube> cubes;
  unsigned numVars;

  SOPForm(unsigned numVars) : numVars(numVars) {}

  /// Compute the truth table represented by this SOP form
  llvm::APInt computeTruthTable() const;

#ifndef NDEBUG
  /// Debug dump method for SOP forms
  void dump(llvm::raw_ostream &os = llvm::errs()) const;
#endif
};

/// Extract ISOP (Irredundant Sum-of-Products) from a truth table.
///
/// Computes an ISOP cover for a Boolean function using the Minato-Morreale
/// algorithm. An ISOP is a sum-of-products where:
///   1. No cube can be removed without changing the function
///   2. The cubes are pairwise disjoint (no minterm is covered by multiple
///   cubes)
///
/// Parameters:
///   truthTable: Single-output truth table represented as APInt
///   numVars: Number of variables (truthTable must have 2^numVars bits)
///
/// Returns: SOPForm representing the ISOP cover
SOPForm extractISOP(const llvm::APInt &truthTable, unsigned numVars);

} // namespace circt

#endif // CIRCT_SUPPORT_TRUTHTABLE_H
