//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offline NPN witness tables for wide (7+ input) library cells.
//
// Load-time pin-permutation enumeration (n! per cell) does not scale per run
// for wide cells, so their entries are precomputed once by an offline
// generator (sharing enumeratePatternEntries, capped per key) and checked in
// here. The pattern set instantiates per-cell entries from this data with
// mechanical verification of every entry, so corruption can never silently
// miscompile.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_WIDEPATTERNTABLES_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_WIDEPATTERNTABLES_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace circt {
namespace synth {
namespace detail {

/// One pin-map occurrence for a wide function: pattern pin p is driven by
/// cut input pinMap[p], with input-negation witness inputNegation.
struct WidePatternEntryData {
  uint16_t pinMap[9];
  uint16_t inputNegation = 0;
};

/// Entries sharing one N-canonical key table.
struct WidePatternKeyGroup {
  unsigned numTableWords = 0;
  const uint64_t *keyTable = nullptr;
  unsigned numEntries = 0;
  const WidePatternEntryData *entries = nullptr;
};

/// One wide function: raw table plus grouped entries.
struct WidePatternFunction {
  const char *name = nullptr;
  unsigned numInputs = 0;
  unsigned numTableWords = 0;
  const uint64_t *rawTable = nullptr;
  unsigned numGroups = 0;
  const WidePatternKeyGroup *groups = nullptr;
};

/// All offline wide functions (empty until data lands).
llvm::ArrayRef<WidePatternFunction> getWidePatternFunctions();

} // namespace detail
} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_WIDEPATTERNTABLES_H
