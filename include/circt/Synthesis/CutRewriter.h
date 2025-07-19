//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines a general cut-based rewriting framework for
// combinational logic optimization. The framework uses NPN-equivalence matching
// with area and delay metrics to rewrite cuts (subgraphs) in combinational
// circuits with optimal patterns.
//
// Applications include:
// - Technology mapping and LUT mapping
// - AIG rewriting and optimization
// - Lazy Man's synthesis
//
// The framework works with any combinational logic representation, though
// And-Inverter Graphs (AIGs) are particularly well-suited.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SYNTHESIS_CUT_REWRITER_H
#define CIRCT_SYNTHESIS_CUT_REWRITER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
#include <optional>

namespace circt {
namespace synthesis {

/// Optimization strategy for cut-based rewriting.
/// Determines whether to prioritize area or timing during rewriting.
enum CutRewriteStrategy {
  Area,  ///< Optimize for minimal area
  Timing ///< Optimize for minimal critical path delay
};

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
struct TruthTable {
  unsigned numInputs;  ///< Number of inputs for this boolean function
  unsigned numOutputs; ///< Number of outputs for this boolean function
  llvm::APInt table;   ///< Truth table data as a packed bit vector

  /// Default constructor creates an empty truth table.
  TruthTable() = default;

  /// Constructor for a truth table with given dimensions and evaluation data.
  TruthTable(unsigned numInputs, unsigned numOutputs, const llvm::APInt &eval)
      : numInputs(numInputs), numOutputs(numOutputs), table(eval) {}

  /// Constructor for a truth table with given dimensions, initialized to zero.
  TruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1u << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination.
  llvm::APInt getOutput(const llvm::APInt &input) const;

  /// Set the output value for a given input combination.
  void setOutput(const llvm::APInt &input, const llvm::APInt &output);

  /// Apply input permutation to create a new truth table.
  /// This reorders the input variables according to the given permutation.
  TruthTable
  applyPermutation(const llvm::SmallVectorImpl<unsigned> &permutation) const;

  /// Apply input negation to create a new truth table.
  /// This negates selected input variables based on the mask.
  TruthTable applyInputNegation(unsigned mask) const;

  /// Apply output negation to create a new truth table.
  /// This negates selected output variables based on the mask.
  TruthTable applyOutputNegation(unsigned negation) const;

  /// Check if this truth table is lexicographically smaller than another.
  /// Used for canonical ordering of truth tables.
  bool isLexicographicallySmaller(const TruthTable &other) const;

  /// Equality comparison for truth tables.
  bool operator==(const TruthTable &other) const;
};

/// Represents the canonical form of a boolean function under NPN equivalence.
///
/// NPN (Negation-Permutation-Negation) equivalence considers two boolean
/// functions equivalent if one can be obtained from the other by:
/// 1. Negating some inputs (pre-negation)
/// 2. Permuting the inputs
/// 3. Negating some outputs (post-negation)
///
/// This canonical form is used to efficiently match cuts against library
/// patterns, as functions in the same NPN class can be implemented by the
/// same circuit with appropriate input/output inversions.
struct NPNClass {
  TruthTable truthTable;                        ///< Canonical truth table
  llvm::SmallVector<unsigned> inputPermutation; ///< Input permutation applied
  unsigned inputNegation = 0;                   ///< Input negation mask
  unsigned outputNegation = 0;                  ///< Output negation mask

  /// Default constructor creates an empty NPN class.
  NPNClass() = default;

  /// Constructor from a truth table.
  NPNClass(const TruthTable &tt) : truthTable(tt) {}

  NPNClass(const TruthTable &tt, llvm::SmallVector<unsigned> inputPerm,
           unsigned inputNeg, unsigned outputNeg)
      : truthTable(tt), inputPermutation(std::move(inputPerm)),
        inputNegation(inputNeg), outputNegation(outputNeg) {}

  /// Compute the canonical NPN form for a given truth table.
  ///
  /// This method exhaustively tries all possible input permutations and
  /// negations to find the lexicographically smallest canonical form.
  ///
  /// WARNING: This is exponential in the number of inputs and should only
  /// be used for small truth tables (< 20 inputs).
  ///
  /// NOTE: This implementation uses exact canonicalization. For larger
  /// truth tables, semi-canonical forms should be used instead.
  static NPNClass computeNPNCanonicalForm(const TruthTable &tt);

  /// Equality comparison for NPN classes.
  bool equivalentOtherThanPermutation(const NPNClass &other) const {
    return truthTable == other.truthTable &&
           inputNegation == other.inputNegation &&
           outputNegation == other.outputNegation;
  }

  bool isLexicographicallySmaller(const NPNClass &other) const {
    if (truthTable.table != other.truthTable.table)
      return truthTable.isLexicographicallySmaller(other.truthTable);
    if (inputNegation != other.inputNegation)
      return inputNegation < other.inputNegation;
    return outputNegation < other.outputNegation;
  }
};

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

// Forward declarations
class CutRewriterPatternSet;
class CutRewriter;
struct CutRewriterPattern;
struct CutRewriterOptions;

/// Represents a cut in the combinational logic network.
///
/// A cut is a subset of nodes in the combinational logic that forms a complete
/// subgraph with a single output. It represents a portion of the circuit that
/// can potentially be replaced with a single library gate or pattern.
///
/// The cut contains:
/// - Input values: The boundary between the cut and the rest of the circuit
/// - Operations: The logic operations within the cut boundary
/// - Root operation: The output-driving operation of the cut
///
/// Cuts are used in combinational logic optimization to identify regions that
/// can be optimized and replaced with more efficient implementations.
class Cut {
  /// Cached truth table for this cut.
  /// Computed lazily when first accessed to avoid unnecessary computation.
  mutable std::optional<mlir::FailureOr<TruthTable>> truthTable;

  /// Cached NPN canonical form for this cut.
  /// Computed lazily from the truth table when first accessed.
  mutable std::optional<mlir::FailureOr<NPNClass>> npnClass;

public:
  /// External inputs to this cut (cut boundary).
  /// These are the values that flow into the cut from outside.
  llvm::SmallSetVector<mlir::Value, 4> inputs;

  /// Operations contained within this cut.
  /// Stored in topological order with the root operation at the end.
  llvm::SmallSetVector<mlir::Operation *, 4> operations;

  /// Check if this cut represents a primary input.
  /// A primary input cut has no internal operations and exactly one input.
  bool isPrimaryInput() const;

  /// Default constructor creates an empty cut.
  Cut() = default;

  /// Get the root operation of this cut.
  /// The root operation produces the output of the cut.
  mlir::Operation *getRoot() const;

  /// Get the NPN canonical form for this cut.
  /// This is used for efficient pattern matching against library components.
  const mlir::FailureOr<NPNClass> &getNPNClass() const;

  void dump() const;

  /// Merge this cut with another cut to form a new cut.
  /// The new cut combines the operations from both cuts with the given root.
  Cut mergeWith(const Cut &other, Operation *root) const;

  /// Get the number of inputs to this cut.
  unsigned getInputSize() const;

  /// Get the number of operations in this cut.
  unsigned getCutSize() const;

  /// Get the number of outputs from root operation.
  size_t getOutputSize() const;

  /// Get the truth table for this cut.
  /// The truth table represents the boolean function computed by this cut.
  const llvm::FailureOr<TruthTable> &getTruthTable() const;

  /// Simulate a single operation for truth table computation.
  /// This method evaluates the operation's logic function with given inputs.
  LogicalResult simulateOp(Operation *op,
                           DenseMap<mlir::Value, llvm::APInt> &values) const;
};

/// Represents a cut that has been successfully matched to a rewriting pattern.
///
/// This class encapsulates the result of matching a cut against a rewriting
/// pattern during optimization. It stores the matched pattern, the
/// cut that was matched, and timing information needed for optimization.
class MatchedPattern {
private:
  CutRewriterPattern *pattern = nullptr; ///< The matched library pattern
  Cut *cut = nullptr;                    ///< The cut that was matched
  double arrivalTime;                    ///< Arrival time through this pattern

public:
  /// Default constructor creates an invalid matched pattern.
  MatchedPattern() = default;

  /// Constructor for a valid matched pattern.
  MatchedPattern(CutRewriterPattern *pattern, Cut *cut, double arrivalTime)
      : pattern(pattern), cut(cut), arrivalTime(arrivalTime) {}

  /// Get the arrival time of signals through this pattern.
  double getArrivalTime() const;

  /// Get the library pattern that was matched.
  CutRewriterPattern *getPattern() const;

  /// Get the cut that was matched to the pattern.
  Cut *getCut() const;

  /// Get the area cost of using this pattern.
  double getArea() const;

  /// Get the delay between specific input and output pins.
  double getDelay(unsigned inputIndex, unsigned outputIndex) const;

  /// Check if this is a valid matched pattern.
  bool isValid() const;
};

/// Manages a collection of cuts for a single logic node using priority cuts
/// algorithm.
///
/// Each node in the combinational logic network can have multiple cuts
/// representing different ways to group it with surrounding logic. The CutSet
/// manages these cuts and selects the best one based on the optimization
/// strategy (area or timing).
///
/// The priority cuts algorithm maintains a bounded set of the most promising
/// cuts to avoid exponential explosion while ensuring good optimization
/// results.
class CutSet {
private:
  llvm::SmallVector<Cut, 12> cuts; ///< Collection of cuts for this node
  std::optional<MatchedPattern> matchedPattern; ///< Best matched pattern found
  bool isFrozen = false; ///< Whether cut set is finalized

public:
  /// Virtual destructor for base class.
  virtual ~CutSet() = default;

  /// Get the best matched pattern for this cut set.
  std::optional<MatchedPattern> getMatchedPattern() const;

  /// Check if this cut set has a valid matched pattern.
  bool isMatched() const;

  /// Get the arrival time of the best matched pattern.
  /// NOTE: isMatched() must be true
  double getArrivalTime() const;

  /// Get the cut associated with the best matched pattern.
  /// NOTE: isMatched() must be true
  Cut *getMatchedCut();

  /// Finalize the cut set by removing duplicates and selecting the best
  /// pattern.
  ///
  /// This method:
  /// 1. Removes duplicate cuts based on inputs and root operation
  /// 2. Limits the number of cuts to prevent exponential growth
  /// 3. Matches each cut against available patterns
  /// 4. Selects the best pattern based on the optimization strategy
  void freezeCutSet(
      const CutRewriterOptions &options,
      llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut);

  /// Get the number of cuts in this set.
  size_t size() const;

  /// Add a new cut to this set.
  /// NOTE: The cut set must not be frozen
  void addCut(Cut cut);

  /// Get read-only access to all cuts in this set.
  ArrayRef<Cut> getCuts() const;
};

/// Base class for cut rewriting patterns used in combinational logic
/// optimization.
///
/// A CutRewriterPattern represents a library component or optimization pattern
/// that can replace cuts in the combinational logic network. Each pattern
/// defines:
/// - How to recognize matching cuts
/// - How to transform/replace the matched cuts
/// - Area and timing characteristics
///
/// Patterns can use truth table matching for efficient recognition or
/// implement custom matching logic for more complex cases.
struct CutRewriterPattern {
  /// Virtual destructor for base class.
  virtual ~CutRewriterPattern() = default;

  /// Check if a cut matches this pattern.
  ///
  /// This method is called to determine if a cut can be replaced by this
  /// pattern. If useTruthTableMatcher() returns true, this method is only
  /// called for cuts with matching truth tables.
  virtual bool match(const Cut &cut) const = 0;

  /// Specify truth tables that this pattern can match.
  ///
  /// If this method returns true, the pattern matcher will use truth table
  /// comparison for efficient pre-filtering. Only cuts with matching truth
  /// tables will be passed to the match() method.
  virtual bool
  useTruthTableMatcher(SmallVectorImpl<NPNClass> &matchingNPNClasses) const;

  /// Rewrite a matched cut by replacing it with this pattern.
  ///
  /// This method performs the actual transformation, replacing the cut's
  /// operations with the pattern's implementation. The cut's root operation
  /// must be replaced or removed if returning success.
  virtual LogicalResult rewrite(PatternRewriter &rewriter,
                                Cut &cutSet) const = 0;

  /// Get the area cost of this pattern.
  virtual double getArea(const Cut &cut) const = 0;

  /// Get the delay between specific input and output pins.
  virtual double getDelay(const Cut &cut, size_t inputIndex,
                          size_t outputIndex) const = 0;

  /// Get the number of inputs this pattern expects.
  virtual unsigned getNumInputs() const = 0;

  /// Get the number of outputs this pattern produces.
  virtual unsigned getNumOutputs() const = 0;

  /// Get the name of this pattern. Used for debugging.
  virtual StringRef getPatternName() const { return "<unnamed>"; }
};

/// Manages a collection of rewriting patterns for combinational logic
/// optimization.
///
/// This class organizes and provides efficient access to rewriting patterns
/// used during cut-based optimization. It maintains:
/// - A collection of all available patterns
/// - Fast lookup tables for truth table-based matching
/// - Separation of truth table vs. custom matching patterns
///
/// The pattern set is used by the CutRewriter to find suitable replacements
/// for cuts in the combinational logic network.
class CutRewriterPatternSet {
public:
  /// Constructor that takes ownership of the provided patterns.
  ///
  /// During construction, patterns are analyzed and organized for efficient
  /// lookup. Truth table matchers are indexed by their NPN canonical forms.
  CutRewriterPatternSet(
      llvm::SmallVector<std::unique_ptr<CutRewriterPattern>, 4> patterns);

private:
  /// Owned collection of all rewriting patterns.
  llvm::SmallVector<std::unique_ptr<CutRewriterPattern>, 4> patterns;

  /// Fast lookup table mapping NPN canonical forms to matching patterns.
  /// Each entry maps an NPN truth table to patterns that can handle it.
  DenseMap<APInt, SmallVector<std::pair<NPNClass, CutRewriterPattern *>>>
      npnToPatternMap;

  /// Patterns that use custom matching logic instead of truth tables.
  /// These patterns are checked against every cut.
  SmallVector<CutRewriterPattern *, 4> nonTruthTablePatterns;

  /// CutRewriter needs access to internal data structures for pattern matching.
  friend class CutRewriter;
};

/// Configuration options for the cut-based rewriting algorithm.
///
/// These options control various aspects of the rewriting process including
/// optimization strategy, resource limits, and algorithmic parameters.
struct CutRewriterOptions {
  /// Optimization strategy (area vs. timing).
  CutRewriteStrategy strategy;

  /// Maximum number of inputs allowed for any cut.
  /// Larger cuts provide more optimization opportunities but increase
  /// computational complexity exponentially.
  unsigned maxCutInputSize;

  /// Maximum number of cuts to maintain per logic node.
  /// The priority cuts algorithm keeps only the most promising cuts
  /// to prevent exponential explosion.
  unsigned maxCutSizePerRoot;
};

class CutEnumerator {
public:
  /// Maps values to their associated cut sets.
  /// Enumerate cuts for all nodes in the given module.
  LogicalResult enumerateCuts(
      Operation *hwModule,
      llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut);
  CutEnumerator(const CutRewriterOptions &options) : options(options) {}

  /// Get the cut set for a specific value.
  CutSet *lookup(Value value) const {
    auto *it = cutSets.find(value);
    if (it != cutSets.end())
      return it->second.get();
    return nullptr;
  }

  CutSet *createNewCutSet(Value value) {
    assert(!cutSets.contains(value) && "Cut set already exists for this value");
    auto cutSet = std::make_unique<CutSet>();
    auto *cutSetPtr = cutSet.get();
    cutSets[value] = std::move(cutSet);
    return cutSetPtr;
  }

  /// Get the cut set for a specific value.
  /// Creates a new cut set if one doesn't exist.
  const CutSet &getCutSet(Value value);

  /// Move and return the cut set for a specific value.

  auto takeVector() { return std::move(cutSets); }

  void clear() { cutSets.clear(); }

private:
  LogicalResult visit(Operation *op);
  LogicalResult visitLogicOp(Operation *logicOp);

  llvm::MapVector<Value, std::unique_ptr<CutSet>> cutSets;
  const CutRewriterOptions &options;

  llvm::function_ref<std::optional<MatchedPattern>(Cut &)>
      matchCut; ///< Function to match cuts against patterns
};

/// Main cut-based rewriting algorithm for combinational logic optimization.
///
/// The CutRewriter implements a cut-based rewriting algorithm that:
/// 1. Enumerates cuts in the combinational logic network using a priority cuts
/// algorithm
/// 2. Matches cuts against available rewriting patterns
/// 3. Selects optimal patterns based on area or timing objectives
/// 4. Rewrites the circuit using the selected patterns
///
/// The algorithm processes the network in topological order, building up cut
/// sets for each node and selecting the best implementation based on the
/// specified optimization strategy.
///
/// Usage example:
/// ```cpp
/// CutRewriterOptions options;
/// options.strategy = CutRewriteStrategy::Area;
/// options.maxCutInputSize = 4;
/// options.maxCutSizePerRoot = 8;
///
/// CutRewriterPatternSet patterns(std::move(optimizationPatterns));
/// CutRewriter rewriter(module, options, patterns);
///
/// if (failed(rewriter.run())) {
///   // Handle rewriting failure
/// }
/// ```
class CutRewriter {
public:
  /// Constructor for the cut rewriter.
  CutRewriter(Operation *op, const CutRewriterOptions &options,
              CutRewriterPatternSet &patterns)
      : topOp(op), options(options), patterns(patterns),
        cutEnumerator(options) {}

  /// Execute the complete cut-based rewriting algorithm.
  ///
  /// This method orchestrates the entire rewriting process:
  /// 1. Enumerate cuts for all nodes in the combinational logic
  /// 2. Match cuts against available patterns
  /// 3. Select optimal patterns based on strategy
  /// 4. Rewrite the circuit with selected patterns
  LogicalResult run();

private:
  /// Enumerate cuts for all nodes in the given module.
  LogicalResult enumerateCuts(Operation *hwModule);

  /// Get the cut set for a specific value.
  /// Creates a new cut set if one doesn't exist.
  const CutSet &getCutSet(Value value);

  /// Get or create a cut set for a specific value.
  CutSet *getOrCreateCutSet(Value value);

  /// Find patterns that match a cut's truth table.
  ArrayRef<std::pair<NPNClass, CutRewriterPattern *>>
  getMatchingPatternFromTruthTable(const Cut &cut) const;

  /// Match a cut against available patterns and compute arrival time.
  std::optional<MatchedPattern> matchCutToPattern(Cut &cut);

  /// Perform the actual circuit rewriting using selected patterns.
  LogicalResult performRewriting(Operation *hwModule);

  Operation *topOp;                      ///< Root operation being rewritten
  const CutRewriterOptions &options;     ///< Configuration options
  const CutRewriterPatternSet &patterns; ///< Available rewriting patterns
  CutEnumerator cutEnumerator;
};

} // namespace synthesis
} // namespace circt

#endif // CIRCT_SYNTHESIS_CUT_REWRITER_H
