//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility structures and functions for the Synth
// dialect, including ISOP (Irredundant Sum-of-Products) extraction.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHUTILS_H
#define CIRCT_DIALECT_SYNTH_SYNTHUTILS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {

// Forward declaration
struct BinaryTruthTable;

namespace synth {

//===----------------------------------------------------------------------===//
// ISOP (Irredundant Sum-of-Products) Extraction
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
  llvm::SmallVector<Cube> cubes;
  unsigned numVars;

  SOPForm(unsigned numVars) : numVars(numVars) {}

  void dump(llvm::raw_ostream &os) const;
  llvm::APInt computeTruthTable() const;
  // Check if the SOP form is irredundant (no cube can be removed).
  // This is slow brute-force check and shouldn't be used in ISOP extraction
  // itself.
  bool isIrredundant();
};

/// Extract ISOP (Irredundant Sum-of-Products) from truth table.
/// Uses Minato-Morreale algorithm for efficient ISOP computation.
///
/// An ISOP is a sum-of-products where:
///   1. No cube can be removed without changing the function
///   2. The cubes are pairwise disjoint (no minterm is covered by multiple
///   cubes)
SOPForm extractSOPFromTruthTable(const APInt &truthTable);

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_SYNTHUTILS_H
