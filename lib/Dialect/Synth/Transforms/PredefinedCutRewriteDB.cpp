//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements predefined cut-rewrite database generation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <numeric>
#include <optional>

using namespace circt;
using namespace circt::synth;
using namespace mlir;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_GENPREDEFINED
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

namespace {

constexpr unsigned kMaxPredefinedTruthTableInputs = 5;
constexpr unsigned kNumNPN4Classes = 222;
constexpr unsigned kNumNPN5Inputs = 5;
constexpr unsigned kNumNPN5Classes = 616126;
constexpr unsigned kNPN5InputNegationCount = 1u << kNumNPN5Inputs;
constexpr unsigned kNPN5OrbitCapacity =
    kNumNPN5Inputs * (kNumNPN5Inputs - 1) * (kNumNPN5Inputs - 2) *
    (kNumNPN5Inputs - 3) * (kNumNPN5Inputs - 4) * kNPN5InputNegationCount;
constexpr uint64_t kNPN5TruthTableHalfSpaceSize =
    1ULL << ((1u << kNumNPN5Inputs) - 1);
constexpr uint64_t kMinCanonicalBucketSize = 1ULL << 10;
constexpr uint64_t kMaxCanonicalBucketSize = 1ULL << 18;

std::string normalizePredefinedDatabaseKind(StringRef kind) {
  std::string normalized = kind.lower();
  for (char &c : normalized)
    if (c == '_')
      c = '-';
  return normalized;
}

void collectCanonicalTruthTablesBruteForce(
    unsigned numInputs, SmallVectorImpl<BinaryTruthTable> &truthTables) {
  truthTables.clear();
  assert(numInputs < 5 &&
         "brute-force predefined generation only supports up to 4 inputs");

  unsigned numTruthTableBits = 1u << numInputs;
  uint64_t numFunctions = 1ULL << numTruthTableBits;
  llvm::BitVector seen(numFunctions, false);

  SmallVector<unsigned> permutation(numInputs);
  std::iota(permutation.begin(), permutation.end(), 0);
  SmallVector<SmallVector<unsigned>> permutations;
  do {
    permutations.push_back(permutation);
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  for (uint64_t value = 0; value != numFunctions; ++value) {
    if (seen.test(value))
      continue;

    BinaryTruthTable seed(numInputs, 1, APInt(numTruthTableBits, value));
    BinaryTruthTable best = seed;

    for (uint32_t negMask = 0; negMask < (1u << numInputs); ++negMask) {
      BinaryTruthTable negatedTT = seed.applyInputNegation(negMask);
      for (ArrayRef<unsigned> perm : permutations) {
        BinaryTruthTable permutedTT = negatedTT.applyPermutation(perm);
        for (unsigned outputNegMask = 0; outputNegMask != 2; ++outputNegMask) {
          BinaryTruthTable candidateTT =
              permutedTT.applyOutputNegation(outputNegMask);
          seen.set(candidateTT.table.getZExtValue());
          if (candidateTT.isLexicographicallySmaller(best))
            best = std::move(candidateTT);
        }
      }
    }

    truthTables.push_back(std::move(best));
  }
}

void setBit(std::vector<std::atomic<uint64_t>> &bitset, uint64_t bit) {
  uint64_t mask = 1ULL << (bit % 64);
  bitset[bit / 64].fetch_or(mask, std::memory_order_relaxed);
}

uint64_t findNextClearBit(const std::vector<std::atomic<uint64_t>> &bitset,
                         uint64_t startBit, uint64_t endBit) {
  if (startBit >= endBit)
    return endBit;

  uint64_t wordIndex = startBit / 64;
  uint64_t bitIndex = startBit % 64;
  uint64_t endWordIndex = (endBit - 1) / 64;
  while (wordIndex <= endWordIndex) {
    uint64_t available =
        ~bitset[wordIndex].load(std::memory_order_relaxed);
    if (wordIndex == startBit / 64)
      available &= ~0ULL << bitIndex;
    if (wordIndex == endWordIndex && (endBit % 64) != 0)
      available &= (1ULL << (endBit % 64)) - 1;
    if (available != 0) {
      uint64_t candidate =
          wordIndex * 64 + static_cast<unsigned>(__builtin_ctzll(available));
      return candidate < endBit ? candidate : endBit;
    }
    ++wordIndex;
    bitIndex = 0;
  }
  return endBit;
}

uint64_t chooseCanonicalBucketSize(MLIRContext *context, uint64_t numValues) {
  uint64_t numThreads =
      context->isMultithreadingEnabled() ? context->getNumThreads() : 1;
  uint64_t targetBuckets = std::max<uint64_t>(numThreads * 16, 1);
  uint64_t bucketSize = (numValues + targetBuckets - 1) / targetBuckets;
  return std::clamp(bucketSize, kMinCanonicalBucketSize,
                    kMaxCanonicalBucketSize);
}

template <typename ValueTy, typename OrbitEnumeratorTy>
void collectCanonicalRepresentativesInBuckets(
    MLIRContext *context, uint64_t numValues, unsigned orbitCapacity,
    unsigned expectedClasses, OrbitEnumeratorTy enumerateOrbit,
    SmallVectorImpl<ValueTy> &representatives) {
  uint64_t bucketSize = chooseCanonicalBucketSize(context, numValues);
  uint64_t numBuckets = (numValues + bucketSize - 1) / bucketSize;
  std::vector<std::atomic<uint64_t>> seen((numValues + 63) / 64);
  for (auto &word : seen)
    word.store(0, std::memory_order_relaxed);

  std::vector<SmallVector<ValueTy, 0>> bucketRepresentatives(numBuckets);
  mlir::parallelFor(context, 0, numBuckets, [&](uint64_t bucketIndex) {
    uint64_t bucketStart = bucketIndex * bucketSize;
    uint64_t bucketEnd = std::min(bucketStart + bucketSize, numValues);
    auto &localRepresentatives = bucketRepresentatives[bucketIndex];
    SmallVector<ValueTy, 0> orbit;
    orbit.reserve(orbitCapacity);

    for (uint64_t candidate = findNextClearBit(seen, bucketStart, bucketEnd);
         candidate != bucketEnd;
         candidate = findNextClearBit(seen, candidate + 1, bucketEnd)) {
      orbit.clear();
      ValueTy best = static_cast<ValueTy>(candidate);
      enumerateOrbit(static_cast<ValueTy>(candidate), [&](ValueTy transformed) {
        orbit.push_back(transformed);
        if (transformed < best)
          best = transformed;
      });

      localRepresentatives.push_back(best);
      for (ValueTy transformed : orbit)
        setBit(seen, transformed);
    }

    llvm::sort(localRepresentatives);
    localRepresentatives.erase(
        std::unique(localRepresentatives.begin(), localRepresentatives.end()),
        localRepresentatives.end());
  });

  representatives.clear();
  size_t totalRepresentatives = 0;
  for (const auto &bucket : bucketRepresentatives)
    totalRepresentatives += bucket.size();
  representatives.reserve(totalRepresentatives);
  for (auto &bucket : bucketRepresentatives)
    representatives.append(bucket.begin(), bucket.end());
  llvm::sort(representatives);
  representatives.erase(
      std::unique(representatives.begin(), representatives.end()),
      representatives.end());
  assert(representatives.size() == expectedClasses &&
         "unexpected number of predefined NPN classes");
}

uint32_t flipTruthTableVariable(uint32_t truthTable, unsigned variable) {
  constexpr std::array<uint32_t, 5> positiveMasks = {
      0xAAAAAAAAu, 0xCCCCCCCCu, 0xF0F0F0F0u, 0xFF00FF00u, 0xFFFF0000u};
  uint32_t positiveMask = positiveMasks[variable];
  uint32_t shift = 1u << variable;
  return ((truthTable & positiveMask) >> shift) |
         ((truthTable & ~positiveMask) << shift);
}

uint32_t swapAdjacentTruthTableVariables(uint32_t truthTable,
                                         unsigned variable) {
  constexpr std::array<std::array<uint32_t, 3>, 4> permutationMasks = {{
      {0x99999999u, 0x22222222u, 0x44444444u},
      {0xC3C3C3C3u, 0x0C0C0C0Cu, 0x30303030u},
      {0xF00FF00Fu, 0x00F000F0u, 0x0F000F00u},
      {0xFF0000FFu, 0x0000FF00u, 0x00FF0000u},
  }};
  uint32_t shift = 1u << variable;
  const auto &masks = permutationMasks[variable];
  return (truthTable & masks[0]) | ((truthTable & masks[1]) << shift) |
         ((truthTable & masks[2]) >> shift);
}

uint32_t normalizeTruthTableOutputPhase(uint32_t truthTable,
                                        unsigned numInputs) {
  unsigned topBit = (1u << numInputs) - 1;
  if ((truthTable >> topBit) & 1u)
    return ~truthTable;
  return truthTable;
}

unsigned getGrayCodeChangedBit(unsigned grayIndex, bool reverse) {
  unsigned currentIndex = reverse ? (kNPN5InputNegationCount - 1 - grayIndex)
                                  : grayIndex;
  unsigned nextIndex = reverse ? (kNPN5InputNegationCount - 2 - grayIndex)
                               : (grayIndex + 1);
  unsigned currentCode = currentIndex ^ (currentIndex >> 1);
  unsigned nextCode = nextIndex ^ (nextIndex >> 1);
  return __builtin_ctz(currentCode ^ nextCode);
}

void enumerateAdjacentPermutations(
    unsigned numInputs,
    function_ref<void(ArrayRef<unsigned>, std::optional<unsigned> swapIndex)>
        callback) {
  SmallVector<unsigned> permutation(numInputs);
  std::iota(permutation.begin(), permutation.end(), 0);
  SmallVector<int> direction(numInputs, -1);

  callback(permutation, std::nullopt);
  while (true) {
    int mobileValue = -1;
    int mobileIndex = -1;
    for (int index = 0, e = static_cast<int>(numInputs); index != e; ++index) {
      int value = static_cast<int>(permutation[index]);
      int nextIndex = index + direction[value];
      if (nextIndex < 0 || nextIndex >= e)
        continue;
      if (value < static_cast<int>(permutation[nextIndex]))
        continue;
      if (value > mobileValue) {
        mobileValue = value;
        mobileIndex = index;
      }
    }
    if (mobileValue < 0)
      return;

    int nextIndex = mobileIndex + direction[mobileValue];
    unsigned swapIndex = static_cast<unsigned>(std::min(mobileIndex, nextIndex));
    std::swap(permutation[mobileIndex], permutation[nextIndex]);
    for (unsigned value = static_cast<unsigned>(mobileValue + 1);
         value != numInputs; ++value)
      direction[value] = -direction[value];
    callback(permutation, swapIndex);
  }
}

struct NPN5OrbitWalker {
  void enumerate(uint32_t truthTable, function_ref<void(uint32_t)> callback) {
    uint32_t current = truthTable;
    unsigned permutationIndex = 0;
    enumerateAdjacentPermutations(
        kNumNPN5Inputs,
        [&](ArrayRef<unsigned>, std::optional<unsigned> swapIndex) {
          if (swapIndex)
            current = swapAdjacentTruthTableVariables(current, *swapIndex);
          enumerateInputNegations(current, permutationIndex, callback);
          ++permutationIndex;
        });
  }

private:
  void enumerateInputNegations(uint32_t &truthTable, unsigned permutationIndex,
                               function_ref<void(uint32_t)> callback) const {
    bool reverseGrayCode = permutationIndex & 1u;
    for (unsigned grayIndex = 0; grayIndex != kNPN5InputNegationCount;
         ++grayIndex) {
      callback(normalizeTruthTableOutputPhase(truthTable, kNumNPN5Inputs));
      if (grayIndex + 1 == kNPN5InputNegationCount)
        break;
      truthTable = flipTruthTableVariable(
          truthTable, getGrayCodeChangedBit(grayIndex, reverseGrayCode));
    }
  }
};

void enumerateNPNTransformOrbit5(uint32_t truthTable,
                                 function_ref<void(uint32_t)> callback) {
  NPN5OrbitWalker walker;
  walker.enumerate(truthTable, callback);
}

void collectCanonicalTruthTables4(MLIRContext *context,
                                  SmallVectorImpl<BinaryTruthTable> &truthTables) {
  SmallVector<uint16_t, kNumNPN4Classes> representatives;
  (void)context;
  collectCanonicalNPN4Representatives(representatives);

  truthTables.clear();
  truthTables.reserve(representatives.size());
  for (uint16_t value : representatives)
    truthTables.emplace_back(4, 1, APInt(16, value));
}

void collectCanonicalTruthTables5(MLIRContext *context,
                                  SmallVectorImpl<BinaryTruthTable> &truthTables) {
  SmallVector<uint32_t, kNumNPN5Classes> representatives;
  collectCanonicalRepresentativesInBuckets<uint32_t>(
      context, kNPN5TruthTableHalfSpaceSize, kNPN5OrbitCapacity,
      kNumNPN5Classes,
      [&](uint32_t truthTable, function_ref<void(uint32_t)> callback) {
        enumerateNPNTransformOrbit5(truthTable, callback);
      },
      representatives);

  truthTables.clear();
  truthTables.reserve(representatives.size());
  for (uint32_t value : representatives)
    truthTables.emplace_back(5, 1, APInt(32, value));
}

void collectCanonicalTruthTables(
    MLIRContext *context, unsigned numInputs,
    SmallVectorImpl<BinaryTruthTable> &truthTables) {
  if (numInputs < 4)
    return collectCanonicalTruthTablesBruteForce(numInputs, truthTables);
  if (numInputs == 4) {
    collectCanonicalTruthTables4(context, truthTables);
    return;
  }

  collectCanonicalTruthTables5(context, truthTables);
}

hw::HWModuleOp createDatabaseEntryModule(
    ModuleOp module, OpBuilder &builder, Builder &attrBuilder,
    StringRef modulePrefix, const BinaryTruthTable &canonicalTT,
    unsigned variantIndex) {
  SmallVector<hw::PortInfo> inputs;
  inputs.reserve(canonicalTT.numInputs);
  for (unsigned i = 0; i != canonicalTT.numInputs; ++i) {
    hw::PortInfo port;
    port.name = attrBuilder.getStringAttr(("i" + Twine(i)).str());
    port.type = builder.getI1Type();
    port.dir = hw::ModulePort::Direction::Input;
    port.argNum = i;
    inputs.push_back(port);
  }

  hw::PortInfo output;
  output.name = attrBuilder.getStringAttr("y");
  output.type = builder.getI1Type();
  output.dir = hw::ModulePort::Direction::Output;
  output.argNum = 0;

  SmallString<64> moduleName;
  moduleName += modulePrefix;
  moduleName += "_i";
  moduleName += Twine(canonicalTT.numInputs).str();
  moduleName += "_tt_";
  SmallString<32> ttString;
  canonicalTT.table.toStringUnsigned(ttString, 16);
  moduleName += ttString;
  moduleName += "_v";
  moduleName += Twine(variantIndex).str();

  return hw::HWModuleOp::create(
      builder, module.getLoc(), attrBuilder.getStringAttr(moduleName),
      hw::ModulePortInfo(inputs, {output}));
}

void appendPredefinedTruthTableEntry(ModuleOp module, OpBuilder &builder,
                                     Builder &attrBuilder,
                                     StringRef modulePrefix,
                                     const BinaryTruthTable &canonicalTT,
                                     unsigned variantIndex) {
  hw::HWModuleOp hwModule = createDatabaseEntryModule(
      module, builder, attrBuilder, modulePrefix, canonicalTT, variantIndex);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(hwModule.getBodyBlock());
  SmallVector<Value> inputs;
  auto arguments = hwModule.getBodyBlock()->getArguments();
  inputs.append(arguments.rbegin(), arguments.rend());

  SmallVector<bool> table;
  table.reserve(canonicalTT.table.getBitWidth());
  for (unsigned i = 0, e = canonicalTT.table.getBitWidth(); i != e; ++i)
    table.push_back(canonicalTT.table[i]);

  Value result =
      comb::TruthTableOp::create(builder, hwModule.getLoc(), inputs, table);
  hwModule.getBodyBlock()->getTerminator()->setOperands({result});
}

LogicalResult populatePredefinedTruthTableDatabase(ModuleOp module,
                                                   StringRef kind,
                                                   unsigned maxInputs) {
  std::string normalizedKind = normalizePredefinedDatabaseKind(kind);
  if (normalizedKind != "npn" && normalizedKind != "npn4") {
    module.emitError() << "unsupported predefined database kind '" << kind
                       << "'";
    return failure();
  }
  if (maxInputs > kMaxPredefinedTruthTableInputs) {
    module.emitError()
        << "predefined NPN generation currently supports at most "
        << kMaxPredefinedTruthTableInputs << " inputs";
    return failure();
  }

  Builder attrBuilder(module.getContext());
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  for (unsigned numInputs = 1; numInputs <= maxInputs; ++numInputs) {
    SmallVector<BinaryTruthTable> canonicalTruthTables;
    collectCanonicalTruthTables(module.getContext(), numInputs,
                                canonicalTruthTables);
    for (const auto &canonicalTT : canonicalTruthTables)
      appendPredefinedTruthTableEntry(module, builder, attrBuilder, "npn",
                                      canonicalTT, 0);
  }
  return success();
}

struct GenPredefinedPass
    : public circt::synth::impl::GenPredefinedBase<GenPredefinedPass> {
  using circt::synth::impl::GenPredefinedBase<
      GenPredefinedPass>::GenPredefinedBase;

  void runOnOperation() override {
    if (failed(
            populatePredefinedTruthTableDatabase(getOperation(), kind, maxInputs)))
      signalPassFailure();
  }
};

} // namespace
