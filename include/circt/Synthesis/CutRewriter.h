//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for CIRCT transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SYNTHESIS_CUT_REWRITER_H
#define CIRCT_SYNTHESIS_CUT_REWRITER_H

#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#define DEBUG_TYPE "synthesis-cut-rewriter"

namespace circt {
namespace synthesis {
// TODO: Add power estimation
enum CutRewriteStrategy { Area, Timing };

struct TruthTable {
  unsigned numInputs;  // Number of inputs for this cut
  unsigned numOutputs; // Number of outputs for this cut
  llvm::APInt table;   // Truth table represented as an llvm::APInt
  TruthTable() = default;
  TruthTable(unsigned numInputs, unsigned numOutputs, const llvm::APInt &eval)
      : numInputs(numInputs), numOutputs(numOutputs), table(eval) {}
  TruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1u << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination
  llvm::APInt getOutput(const llvm::APInt &input) const {
    assert(input.getBitWidth() == numInputs && "Input width mismatch");
    return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
  }

  /// Set the output value for a given input combination
  void setOutput(const llvm::APInt &input, const llvm::APInt &output) {
    assert(input.getBitWidth() == numInputs && "Input width mismatch");
    assert(output.getBitWidth() == numOutputs && "Output width mismatch");
    unsigned offset = input.getZExtValue() * numOutputs;
    for (unsigned i = 0; i < numOutputs; ++i)
      table.setBitVal(offset + i, output[i]);
  }

  /// Apply input permutation to the truth table
  TruthTable
  applyPermutation(const llvm::SmallVectorImpl<unsigned> &permutation) const {
    assert(permutation.size() == numInputs && "Permutation size mismatch");
    TruthTable result(numInputs, numOutputs);

    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      llvm::APInt input(numInputs, i);
      llvm::APInt permutedInput(numInputs, 0);

      // Apply permutation
      for (unsigned j = 0; j < numInputs; ++j)
        permutedInput.setBitVal(j, input[permutation[j]]);

      result.setOutput(permutedInput, getOutput(input));
    }

    return result;
  }

  /// Apply input negation to the truth table
  TruthTable applyInputNegation(unsigned mask) const {
    TruthTable result(numInputs, numOutputs);

    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      llvm::APInt input(numInputs, i);
      llvm::APInt negatedInput(numInputs, 0);

      // Apply negation
      for (unsigned j = 0; j < numInputs; ++j)
        negatedInput.setBitVal(j, (mask & (1u << j)) ? !input[j] : input[j]);

      result.setOutput(negatedInput, getOutput(input));
    }

    return result;
  }

  /// Apply output negation to the truth table
  TruthTable applyOutputNegation(unsigned negation) const {
    TruthTable result(numInputs, numOutputs);

    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      llvm::APInt input(numInputs, i);
      llvm::APInt output = getOutput(input);
      llvm::APInt negatedOutput(numOutputs, 0);

      // Apply negation
      for (unsigned j = 0; j < numOutputs; ++j)
        negatedOutput.setBitVal(j, (negation & (1u << j)) ? !output[j]
                                                          : output[j]);

      result.setOutput(input, negatedOutput);
    }

    return result;
  }

  /// Check if this truth table is lexicographically smaller than another
  bool isLexicographicallySmaller(const TruthTable &other) const {
    assert(numInputs == other.numInputs && numOutputs == other.numOutputs);
    return table.ult(other.table);
  }

  bool operator==(const TruthTable &other) const {
    return numInputs == other.numInputs && numOutputs == other.numOutputs &&
           table == other.table;
  }
};

struct NPNClass {
  TruthTable truthTable;
  llvm::SmallVector<unsigned> inputPermutation;
  unsigned inputNegation = 0;
  unsigned outputNegation = 0;
  NPNClass() = default;
  NPNClass(const TruthTable &tt) : truthTable(tt) {}

  /// Compute "exact" NPN canonical form. This is not practical for large
  /// truth tables.
  /// TODO: Use semi-canonical NPN form instead.
  static NPNClass computeNPNCanonicalForm(const TruthTable &tt) {
    NPNClass canonical(tt);

    // Initialize permutation and negation vectors
    canonical.inputPermutation.resize(tt.numInputs);

    // Initialize identity permutation
    for (unsigned i = 0; i < tt.numInputs; ++i)
      canonical.inputPermutation[i] = i;

    TruthTable bestTT = tt;

    // Try all possible input negations (2^n combinations)
    assert(tt.numInputs <= 20 && "Too many inputs for input negation mask");
    for (uint32_t negMask = 0; negMask < (1u << tt.numInputs); ++negMask) {
      TruthTable negatedTT = tt.applyInputNegation(negMask);

      // Try all possible permutations
      llvm::SmallVector<unsigned> permutation(tt.numInputs);
      for (uint32_t i = 0; i < tt.numInputs; ++i) {
        permutation[i] = i;
      }

      do {
        TruthTable permutedTT = negatedTT.applyPermutation(permutation);

        // Try output negation (for single output)
        if (tt.numOutputs == 1) {
          unsigned outputNegMask = 0;
          TruthTable candidate = permutedTT;

          // Try without output negation
          if (candidate.isLexicographicallySmaller(bestTT)) {
            bestTT = candidate;
            canonical.truthTable = candidate;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = negMask;
            canonical.outputNegation = outputNegMask;
          }

          // Try with output negation
          candidate = permutedTT.applyOutputNegation(1);
          if (candidate.isLexicographicallySmaller(bestTT)) {
            bestTT = candidate;
            canonical.truthTable = candidate;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = negMask;
            canonical.outputNegation = 1;
          }
        } else {
          // For multi-output, just check without output negation
          if (permutedTT.isLexicographicallySmaller(bestTT)) {
            bestTT = permutedTT;
            canonical.truthTable = permutedTT;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = negMask;
          }
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }

    return canonical;
  }
};

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

class CutRewriterPatternSet;
class CutRewriter;
struct CutRewriterPattern;
struct CutRewriterOptions;

/// Represents a cut in the AIG network
struct Cut {
  llvm::SmallSetVector<mlir::Value, 4> inputs; // Cut inputs.
  // Operations in the cut. Operation is topologcally sorted, with the root
  // operation at the back.
  llvm::SmallSetVector<mlir::Operation *, 4> operations;

  bool isPrimaryInput() const {
    // A cut is a primary input if it has no operations and only one input
    return operations.empty() && inputs.size() == 1;
  }

  Cut() = default;

  mlir::Operation *getRoot() const {
    return operations.empty()
               ? nullptr
               : operations.back(); // The first operation is the root
  }

  // Cache for the truth table of this cut
  mutable std::optional<mlir::FailureOr<TruthTable>> truthTable;
  // Compute the NPN canonical form for this cut
  mutable std::optional<mlir::FailureOr<NPNClass>> npnClass;

  const mlir::FailureOr<NPNClass> &getNPNClass() const {
    // If the truth table is already computed, return it
    if (npnClass)
      return *npnClass;

    // Compute the NPN (Negation Permutation Negation) canonical form for this
    // cut
    auto truthTable = getTruthTable();
    assert(succeeded(truthTable) && "Failed to compute truth table");

    // Compute the NPN canonical form
    auto canonicalForm = NPNClass::computeNPNCanonicalForm(*truthTable);

    npnClass.emplace(std::move(canonicalForm));
    return *npnClass;
  }

  void dump() const {
    llvm::dbgs() << "==========================\n";
    llvm::dbgs() << "Cut with " << getInputSize() << " inputs and "
                 << getCutSize() << " operations:\n";
    if (isPrimaryInput()) {
      llvm::dbgs() << "Primary input cut: " << *inputs.begin() << "\n";
      return;
    }

    llvm::dbgs() << "Inputs: ";
    for (auto [idx, input] : llvm::enumerate(inputs)) {
      llvm::dbgs() << "Input " << idx << ": " << input << "\n";
    }
    llvm::dbgs() << "\nOperations: ";
    for (auto *op : operations) {
      op->dump();
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "Truth Table: ";
    auto truthTable = getTruthTable();
    llvm::dbgs() << truthTable->table << "\n";
    llvm::dbgs() << "NPN Class: ";
    auto &npnClass = getNPNClass();
    llvm::dbgs() << npnClass->truthTable.table << "\n";

    llvm::dbgs() << "==========================\n";
  }

public:
  /// Merge this cut with another cut
  Cut mergeWith(const Cut &other, Operation *root) const;

  /// Get the size of this cut
  unsigned getInputSize() const { return inputs.size(); }
  unsigned getCutSize() const { return operations.size(); }
  size_t getOutputSize() const { return getRoot()->getNumResults(); }

  const llvm::FailureOr<TruthTable> &getTruthTable() const {
    assert(!isPrimaryInput() && "Primary input cuts do not have truth tables");

    if (truthTable)
      return *truthTable;

    int64_t numInputs = getInputSize();
    int64_t numOutputs = getOutputSize();
    assert(numInputs < 20 && "Truth table is too large");

    // Simulate the IR.
    uint32_t tableSize = 1 << numInputs;
    DenseMap<Value, APInt> eval;
    for (uint32_t i = 0; i < numInputs; ++i) {
      APInt value(tableSize, 0);
      for (uint32_t j = 0; j < tableSize; ++j) {
        // Make sure the order of the bits is correct.
        value.setBitVal(j, (j >> i) & 1);
      }
      // Set the input value for the truth table
      eval[inputs[i]] = std::move(value);
    }

    // Simulate the operations in the cut
    for (auto *op : operations) {
      auto result = simulateOp(op, eval);
      if (failed(result)) {
        op->emitError("Failed to simulate operation");
        truthTable = failure();
        return *truthTable;
      }
      // Set the output value for the truth table
      for (size_t j = 0; j < op->getNumResults(); ++j) {
        auto outputValue = op->getResult(j);
        if (!outputValue.getType().isInteger(1)) {
          op->emitError("Output value is not a single bit: ") << outputValue;
          truthTable = failure();
          return *truthTable;
        }
      }
    }

    // Extract the truth table from the root operation
    auto rootResults = getRoot()->getResults();
    assert(rootResults.size() == 1 &&
           "For now we only support single output cuts");
    auto result = rootResults[0];

    // Cache the truth table
    truthTable = TruthTable(numInputs, numOutputs, eval[result]);
    return *truthTable;
  }

  LogicalResult simulateOp(Operation *op,
                           DenseMap<mlir::Value, llvm::APInt> &values) const;
};

class MatchedPattern {
  CutRewriterPattern *pattern = nullptr; // Matched pattern
  Cut *cut = nullptr;                    // Cut that matched the pattern
  double arrivalTime;                    // Arrival time of the cut

public:
  MatchedPattern() = default;
  MatchedPattern(CutRewriterPattern *pattern, Cut *cut, double arrivalTime)
      : pattern(pattern), cut(cut), arrivalTime(arrivalTime) {}

  double getArrivalTime() const {
    assert(pattern && "Pattern must be set to get arrival time");
    return arrivalTime;
  }

  CutRewriterPattern *getPattern() const {
    assert(pattern && "Pattern must be set to get the pattern");
    return pattern;
  }

  Cut *getCut() const {
    assert(cut && "Cut must be set to get the cut");
    return cut;
  }

  double getArea() const;
  double getDelay(unsigned inputIndex, unsigned outputIndex) const;

  bool isValid() const { return pattern != nullptr; }
};

/// Cut set for a node using priority cuts algorithm
class CutSet {
private:
  llvm::SmallVector<Cut, 12> cuts;
  std::optional<MatchedPattern>
      matchedPattern; // Matched pattern for this cut set

  bool isFrozen = false; // Whether the cut set is frozen

public:
  virtual ~CutSet() = default;

  bool isMatched() const {
    return matchedPattern.has_value() && matchedPattern->isValid();
  }

  double getArrivalTime() const {
    assert(isMatched() &&
           "Matched pattern must be set before getting arrival time");
    return matchedPattern->getArrivalTime();
  }

  std::optional<MatchedPattern> getMatchedPattern() const {
    return matchedPattern;
  }

  Cut *getMatchedCut() {
    assert(isMatched() &&
           "Matched pattern must be set before getting matched cut");
    return matchedPattern->getCut();
  }

  // Keep the cuts sorted by priority. The priority is determined by the
  // optimization mode (area or timing). The cuts are sorted in descending
  // order. This also removes duplicate cuts based on their inputs and root
  // operation.
  void freezeCutSet(
      const CutRewriterOptions &options,
      llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut);

  size_t size() const { return cuts.size(); }

  /// Add a cut to the set.
  void addCut(Cut cut) {
    assert(!isFrozen && "Cannot add cuts to a frozen cut set");
    cuts.push_back(std::move(cut));
  }

  /// Iterator that returns all the cuts.
  /// This is a range-based for loop compatible iterator.
  ArrayRef<Cut> getCuts() const { return cuts; }
};

// Base class for Cut rewriting patterns.
struct CutRewriterPattern {
  virtual ~CutRewriterPattern() = default;

  // Return true if the cut matches this cut rewriter pattern. If
  // `useTruthTableMatcher` is implemented to return true, this method is
  // called only when the truth table matches the cut set.
  virtual bool match(const Cut &cut) const = 0;

  // If the pattern can be matched using a truth table, populate `truthTable`
  // with the truth table this pattern potentially matches. If false is
  // returned, the pattern is always tried to match against the cut set.
  virtual bool
  useTruthTableMatcher(SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
    return false;
  }

  // Rewrite the cut. If success is returned, the cut root operation must be
  // replaced with other operation. `match` is called before this method,
  // so the cut set is guaranteed to match this pattern.
  // NOTE: There is no "matchAndRewrite" method, because delay could be
  // non-uniform so first enumerate all possible patterns select based on the
  // storategy.
  virtual LogicalResult rewrite(PatternRewriter &rewriter,
                                Cut &cutSet) const = 0;

  // Get the area after the rewrite. `cut` is guaranteed to match this pattern.
  virtual double getArea(const Cut &cut) const = 0;

  // Get the delay between two specific ports. `cut` is guaranteed to match this
  // pattern.
  virtual double getDelay(const Cut &cut, size_t inputIndex,
                          size_t outputIndex) const = 0;

  virtual unsigned getNumInputs() const = 0;
  virtual unsigned getNumOutputs() const = 0;
};

class CutRewriterPatternSet {
public:
  CutRewriterPatternSet(
      llvm::SmallVector<std::unique_ptr<CutRewriterPattern>, 4> patterns)
      : patterns(std::move(patterns)) {
    // Initialize the NPN to pattern map
    for (auto &pattern : this->patterns) {
      SmallVector<NPNClass, 2> npnClasses;
      if (pattern->useTruthTableMatcher(npnClasses)) {
        for (auto npnClass : npnClasses) {
          // Create a NPN class from the truth table
          npnToPatternMap[npnClass.truthTable.table].push_back(
              std::make_pair(std::move(npnClass), pattern.get()));
        }
      } else {
        // If the pattern does not use truth table matcher, add it to the
        // non-truth table patterns
        nonTruthTablePatterns.push_back(pattern.get());
      }
    }
  }

private:
  // Owns a set of patterns for cut rewriting
  llvm::SmallVector<std::unique_ptr<CutRewriterPattern>, 4> patterns;

  // Map NPN's truth table to NPN classes and pattern.
  DenseMap<APInt, SmallVector<std::pair<NPNClass, CutRewriterPattern *>>>
      npnToPatternMap;
  SmallVector<CutRewriterPattern *, 4> nonTruthTablePatterns;

  friend class CutRewriter;
};

struct CutRewriterOptions {
  CutRewriteStrategy strategy; // Mapping strategy (area or timing)
  unsigned maxCutInputSize;    // Maximum cut input size
  unsigned maxCutSizePerRoot;  // Maximum cut size per root operation
};

/// Main mapper class
class CutRewriter {
public:
  CutRewriter(Operation *op, const CutRewriterOptions &options,
              CutRewriterPatternSet &patterns)
      : topOp(op), options(options), patterns(patterns) {}

  /// Run the mapping algorithm
  LogicalResult run();

private:
  /// Enumerate cuts for all nodes in the module
  LogicalResult enumerateCuts(Operation *hwModule);

  LogicalResult generateCutsForAndOp(Operation *op);

  /// Get the cut set for a specific operation
  const CutSet &getCutSet(Value value);
  CutSet *getOrCreateCutSet(Value value);

  ArrayRef<std::pair<NPNClass, CutRewriterPattern *>>
  getMatchingPatternFromTruthTable(const Cut &cut) const;

  // Compute arrival time to this cut.
  std::optional<MatchedPattern> matchCutToPattern(Cut &cut);

  /// Perform the actual technology mapping
  LogicalResult performMapping(Operation *hwModule);

  Operation *topOp;
  const CutRewriterOptions &options;
  const CutRewriterPatternSet &patterns;
  llvm::MapVector<Value, std::unique_ptr<CutSet>> cutSets;
};
} // namespace synthesis
} // namespace circt
#endif // CIRCT_SYNTHESIS_CUT_REWRITER_H
