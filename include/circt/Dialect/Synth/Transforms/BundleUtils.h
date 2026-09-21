//===- BundleUtils.h - Shared multi-output matching helpers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small truth-table helpers shared by bundle extraction and mapping, over
// plain uint64_t bit vectors (LSB-first rows, sufficient for the small
// supports multi-output matching considers).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_BUNDLEUTILS_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_BUNDLEUTILS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>

namespace circt {
namespace synth {
namespace bundle {

// Reference functions (LSB-first rows) for adder-shaped classification.
constexpr uint64_t kAnd2Table = 0b1000;
constexpr uint64_t kXor2Table = 0b0110;
constexpr uint64_t kMaj3Table = 0b11101000;
constexpr uint64_t kXor3Table = 0b10010110;
constexpr uint64_t kMinority3Table = 0b00010111;

enum class GateClass { None, And2, Xor2, Maj3, Xor3, Minority3 };

/// Apply an input permutation to a truth table: bit `i` of each row moves
/// to position `perm[i]`. Brute force is trivial for the small supports
/// considered here and avoids any canonicalization dependency.
inline uint64_t permuteTableBits(uint64_t table, unsigned numInputs,
                                 ArrayRef<unsigned> perm) {
  uint64_t result = 0;
  for (unsigned row = 0, rows = 1u << numInputs; row < rows; ++row) {
    if ((table >> row) & 1u) {
      unsigned refRow = 0;
      for (unsigned i = 0; i < numInputs; ++i)
        if ((row >> i) & 1u)
          refRow |= 1u << perm[i];
      result |= 1ULL << refRow;
    }
  }
  return result;
}

/// Check whether `table` implements `reference` under some input
/// permutation (positive phases only; negations are left to the discrete
/// phase-aware flow).
inline bool matchesUnderPermutation(uint64_t table, unsigned numInputs,
                                     uint64_t reference) {
  SmallVector<unsigned, 4> perm;
  for (unsigned i = 0; i < numInputs; ++i)
    perm.push_back(i);
  // Single input: exactly one permutation.
  do {
    if (permuteTableBits(reference, numInputs, perm) == table)
      return true;
  } while (std::next_permutation(perm.begin(), perm.end()));
  return false;
}

inline GateClass classifyTable(uint64_t table, unsigned numInputs) {
  if (numInputs == 2) {
    if (matchesUnderPermutation(table, 2, kAnd2Table))
      return GateClass::And2;
    if (matchesUnderPermutation(table, 2, kXor2Table))
      return GateClass::Xor2;
  } else if (numInputs == 3) {
    if (matchesUnderPermutation(table, 3, kMaj3Table))
      return GateClass::Maj3;
    if (matchesUnderPermutation(table, 3, kXor3Table))
      return GateClass::Xor3;
    // Complemented carry: Wallace trees predominantly materialize
    // carries in inverted form, so XOR3+Minority3 is the common
    // full-adder shape (e.g. ASAP7 FAx1 with its CON output).
    if (matchesUnderPermutation(table, 3, kMinority3Table))
      return GateClass::Minority3;
  }
  return GateClass::None;
}

} // namespace bundle
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_BUNDLEUTILS_H
