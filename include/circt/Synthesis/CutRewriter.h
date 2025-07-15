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
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
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
  llvm::APInt getOutput(const llvm::APInt &input) const;

  /// Set the output value for a given input combination
  void setOutput(const llvm::APInt &input, const llvm::APInt &output);

  /// Apply input permutation to the truth table
  TruthTable
  applyPermutation(const llvm::SmallVectorImpl<unsigned> &permutation) const;

  /// Apply input negation to the truth table
  TruthTable applyInputNegation(unsigned mask) const;

  /// Apply output negation to the truth table
  TruthTable applyOutputNegation(unsigned negation) const;

  /// Check if this truth table is lexicographically smaller than another
  bool isLexicographicallySmaller(const TruthTable &other) const;

  bool operator==(const TruthTable &other) const;
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
  static NPNClass computeNPNCanonicalForm(const TruthTable &tt);
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

  bool isPrimaryInput() const;

  Cut() = default;

  mlir::Operation *getRoot() const;

  // Cache for the truth table of this cut
  mutable std::optional<mlir::FailureOr<TruthTable>> truthTable;
  // Compute the NPN canonical form for this cut
  mutable std::optional<mlir::FailureOr<NPNClass>> npnClass;

  const mlir::FailureOr<NPNClass> &getNPNClass() const;

  void dump() const;

public:
  /// Merge this cut with another cut
  Cut mergeWith(const Cut &other, Operation *root) const;

  /// Get the size of this cut
  unsigned getInputSize() const;
  unsigned getCutSize() const;
  size_t getOutputSize() const;

  const llvm::FailureOr<TruthTable> &getTruthTable() const;

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

  double getArrivalTime() const;

  CutRewriterPattern *getPattern() const;

  Cut *getCut() const;

  double getArea() const;
  double getDelay(unsigned inputIndex, unsigned outputIndex) const;

  bool isValid() const;
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

  bool isMatched() const;

  double getArrivalTime() const;

  std::optional<MatchedPattern> getMatchedPattern() const;

  Cut *getMatchedCut();

  // Keep the cuts sorted by priority. The priority is determined by the
  // optimization mode (area or timing). The cuts are sorted in descending
  // order. This also removes duplicate cuts based on their inputs and root
  // operation.
  void freezeCutSet(
      const CutRewriterOptions &options,
      llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut);

  size_t size() const;

  /// Add a cut to the set.
  void addCut(Cut cut);

  /// Iterator that returns all the cuts.
  /// This is a range-based for loop compatible iterator.
  ArrayRef<Cut> getCuts() const;
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
  useTruthTableMatcher(SmallVectorImpl<NPNClass> &matchingNPNClasses) const;

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
      llvm::SmallVector<std::unique_ptr<CutRewriterPattern>, 4> patterns);

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
