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
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H

#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/TruthTable.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <optional>

namespace circt {
namespace synth {
// Type for representing delays in the circuit. It's user's responsibility to
// use consistent units, i.e., all delays should be in the same unit (usually
// femtoseconds, but not limited to it).
using DelayType = int64_t;

/// Maximum number of inputs supported for truth table generation.
/// This limit prevents excessive memory usage as truth table size grows
/// exponentially with the number of inputs (2^n entries).
static constexpr unsigned maxTruthTableInputs = 16;

// This is a helper function to sort operations topologically in a logic
// network. This is necessary for cut rewriting to ensure that operations are
// processed in the correct order, respecting dependencies.
LogicalResult topologicallySortLogicNetwork(mlir::Operation *op);

// Get the truth table for a specific operation within a block.
// Block must be a SSACFG or topologically sorted.
FailureOr<BinaryTruthTable> getTruthTable(ValueRange values, Block *block);

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

// Forward declarations
class CutRewritePatternSet;
class CutRewriter;
class CutEnumerator;
struct CutRewritePattern;
struct CutRewriterOptions;

//===----------------------------------------------------------------------===//
// Internal Implementation Details
//
// Types in the detail namespace are internal implementation details.
// External users should not use these directly.
//===----------------------------------------------------------------------===//

namespace detail {
// Forward declarations - full definitions in CutRewriter.cpp
struct Signal;
struct LogicNetworkGate;
class LogicNetwork;
} // namespace detail

//===----------------------------------------------------------------------===//
///
/// This structure contains the area and per-input delay information
/// computed during pattern matching.
///
/// The delays can be stored in two ways:
/// 1. As a reference to static/cached data (e.g., tech library delays)
///    - Use setDelayRef() for zero-cost reference (no allocation)
/// 2. As owned dynamic data (e.g., computed SOP delays)
///    - Use setOwnedDelays() to transfer ownership
///
struct MatchResult {
  /// Area cost of implementing this cut with the pattern.
  double area = 0.0;

  /// Default constructor.
  MatchResult() = default;

  /// Constructor with area and delays (by reference).
  MatchResult(double area, ArrayRef<DelayType> delays)
      : area(area), borrowedDelays(delays) {}

  /// Set delays by reference (zero-cost for static/cached delays).
  /// The caller must ensure the referenced data remains valid.
  void setDelayRef(ArrayRef<DelayType> delays) { borrowedDelays = delays; }

  /// Set delays by transferring ownership (for dynamically computed delays).
  /// This moves the data into internal storage.
  void setOwnedDelays(SmallVector<DelayType, 6> delays) {
    ownedDelays.emplace(std::move(delays));
    borrowedDelays = {};
  }

  /// Get all delays as an ArrayRef.
  ArrayRef<DelayType> getDelays() const {
    return ownedDelays.has_value() ? ArrayRef<DelayType>(*ownedDelays)
                                   : borrowedDelays;
  }

private:
  /// Borrowed delays (used when ownedDelays is empty).
  /// Points to external data provided via setDelayRef().
  ArrayRef<DelayType> borrowedDelays;

  /// Owned delays (used when present).
  /// Only allocated when setOwnedDelays() is called. When empty (std::nullopt),
  /// moving this MatchResult avoids constructing/moving the SmallVector,
  /// achieving zero-cost abstraction for the common case (borrowed delays).
  std::optional<SmallVector<DelayType, 6>> ownedDelays;
};

/// Represents a cut that has been successfully matched to a rewriting pattern.
///
/// This class encapsulates the result of matching a cut against a rewriting
/// pattern during optimization. It stores the matched pattern, the
/// cut that was matched, and timing information needed for optimization.
class MatchedPattern {
private:
  const CutRewritePattern *pattern = nullptr; ///< The matched library pattern
  SmallVector<DelayType, 1>
      arrivalTimes;  ///< Arrival times of outputs from this pattern
  double area = 0.0; ///< Area cost of this pattern

public:
  /// Default constructor creates an invalid matched pattern.
  MatchedPattern() = default;

  /// Constructor for a valid matched pattern.
  MatchedPattern(const CutRewritePattern *pattern,
                 SmallVector<DelayType, 1> arrivalTimes, double area)
      : pattern(pattern), arrivalTimes(std::move(arrivalTimes)), area(area) {}

  /// Get the arrival time of signals through this pattern.
  DelayType getArrivalTime(unsigned outputIndex) const;
  ArrayRef<DelayType> getArrivalTimes() const;
  DelayType getWorstOutputArrivalTime() const;

  /// Get the library pattern that was matched.
  const CutRewritePattern *getPattern() const;

  /// Get the area cost of using this pattern.
  double getArea() const;
};

/// Represents a cut in the combinational logic network.
///
/// A cut is a subset of nodes in the combinational logic that forms a complete
/// subgraph with a single output. It represents a portion of the circuit that
/// can potentially be replaced with a single library gate or pattern.
///
/// The cut contains:
/// - Input indices: LogicNetwork indices of the boundary values
/// - Root index: LogicNetwork index of the output-driving node
///
/// Cuts are used in combinational logic optimization to identify regions that
/// can be optimized and replaced with more efficient implementations.
///
/// All indices refer to positions in the LogicNetwork, enabling efficient
/// index-based operations without Value comparisons.
class Cut {
  /// Cached truth table for this cut.
  /// Computed lazily when first accessed to avoid unnecessary computation.
  /// Using optional allows us to defer expensive truth table computation
  /// until after duplicate removal, avoiding wasted work.
  mutable std::optional<BinaryTruthTable> truthTable;

  /// Cached NPN canonical form for this cut.
  /// Computed lazily from the truth table when first accessed.
  mutable std::optional<NPNClass> npnClass;

  std::optional<MatchedPattern> matchedPattern;

  /// Root index in LogicNetwork (0 for trivial cuts, which use constant index).
  /// The root node produces the output of the cut.
  uint32_t rootIndex = 0;

  /// Signature bitset for fast cut size estimation.
  /// Bit i is set if value with index i is in the cut's inputs.
  /// This enables O(1) estimation of merged cut size using popcount.
  uint64_t signature = 0;

  /// Operand cuts used to create this cut (for lazy TT computation).
  /// Stored to enable fast incremental truth table computation after
  /// duplicate removal. Using raw pointers is safe since cuts are allocated
  /// via bump allocator and live for the duration of enumeration.
  llvm::SmallVector<const Cut *, 3> operandCuts;

  /// External inputs to this cut (cut boundary).
  /// Stored as LogicNetwork indices for efficient operations.
  llvm::SmallVector<uint32_t, 6> inputs;

public:
  //===--------------------------------------------------------------------===//
  // Public API for CutRewritePattern implementations
  //===--------------------------------------------------------------------===//

  /// Get an input value by index.
  /// \param idx The index into the inputs array (0 to getInputSize()-1)
  /// \param network The LogicNetwork to resolve the value from
  /// \return The MLIR value for the input at the given index
  Value getInputValue(unsigned idx, const LogicNetwork &network) const;

  /// Get all input values for this cut.
  /// \param network The LogicNetwork to resolve values from
  /// \return A SmallVector containing all input values
  SmallVector<Value> getInputValues(const LogicNetwork &network) const;

  /// Get the root operation value.
  /// \param network The LogicNetwork to resolve the value from
  /// \return The MLIR value produced by the root operation
  Value getRootValue(const LogicNetwork &network) const;

  /// Get the root operation.
  /// \param network The LogicNetwork to resolve the operation from
  /// \return The operation at the root of this cut, or nullptr for trivial cuts
  Operation *getRootOperation(const LogicNetwork &network) const;

  /// Check if this cut represents a trivial cut.
  /// A trivial cut has no root operation and exactly one input.
  bool isTrivialCut() const;

  /// Get the number of inputs to this cut.
  unsigned getInputSize() const;

  /// Get the number of outputs from root operation.
  unsigned getOutputSize(const LogicNetwork &network) const;

  /// Get the truth table for this cut.
  /// The truth table represents the boolean function computed by this cut.
  const std::optional<BinaryTruthTable> &getTruthTable() const {
    return truthTable;
  }

  /// Get the NPN canonical form for this cut.
  /// This is used for efficient pattern matching against library components.
  const NPNClass &getNPNClass() const;

  /// Get the permutated inputs for this cut based on the given pattern NPN.
  /// Returns indices into the inputs vector.
  void
  getPermutatedInputIndices(const NPNClass &patternNPN,
                            SmallVectorImpl<unsigned> &permutedIndices) const;

  /// Get arrival times for each input of this cut.
  /// Returns failure if any input doesn't have a valid matched pattern.
  LogicalResult getInputArrivalTimes(CutEnumerator &enumerator,
                                     SmallVectorImpl<DelayType> &results) const;

  /// Dump this cut for debugging.
  void dump(llvm::raw_ostream &os, const LogicNetwork &network) const;

private:
  //===--------------------------------------------------------------------===//
  // Internal implementation - not for external use
  //===--------------------------------------------------------------------===//

  friend class CutEnumerator;
  friend class CutSet;

  /// Get the root index in the LogicNetwork (internal use only).
  uint32_t getRootIndex() const { return rootIndex; }

  /// Set the root index of this cut (internal use only).
  void setRootIndex(uint32_t idx) { rootIndex = idx; }

  /// Get the signature of this cut.
  uint64_t getSignature() const { return signature; }

  /// Set the signature of this cut.
  void setSignature(uint64_t sig) { signature = sig; }

  /// Compute and set the signature based on current inputs.
  /// Uses input indices directly (no ValueNumbering needed).
  void computeSignature();

  /// Estimate the size of merging this cut with another using signatures.
  /// Returns the estimated number of unique inputs in the merged cut.
  unsigned estimateMergedSize(const Cut &other) const;

  /// Compute and cache the truth table for this cut using the LogicNetwork.
  /// Uses slow simulation-based approach - prefer computeTruthTableFromOperands
  /// when operand cuts are available.
  void computeTruthTable(const LogicNetwork &network);

  /// Compute truth table using fast incremental method from operand cuts.
  /// This is much faster than simulation-based computation.
  /// Requires that operand cuts have already been set via setOperandCuts.
  void computeTruthTableFromOperands(const LogicNetwork &network);

  /// Set the truth table directly (used for incremental computation).
  void setTruthTable(BinaryTruthTable tt) { truthTable.emplace(std::move(tt)); }

  /// Set operand cuts for lazy truth table computation.
  /// These are stored to enable fast incremental TT computation later.
  void setOperandCuts(ArrayRef<const Cut *> cuts) {
    operandCuts.assign(cuts.begin(), cuts.end());
  }

  /// Get operand cuts (for fast TT computation).
  ArrayRef<const Cut *> getOperandCuts() const { return operandCuts; }

  /// Matched pattern for this cut.
  void setMatchedPattern(MatchedPattern pattern) {
    matchedPattern = std::move(pattern);
  }

  /// Get the matched pattern for this cut.
  const std::optional<MatchedPattern> &getMatchedPattern() const {
    return matchedPattern;
  }
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
  llvm::SmallVector<Cut *, 4> cuts; ///< Collection of cuts for this node
  Cut *bestCut = nullptr;
  bool isFrozen = false; ///< Whether cut set is finalized

public:
  /// Check if this cut set has a valid matched pattern.
  bool isMatched() const { return bestCut; }

  /// Get the cut associated with the best matched pattern.
  Cut *getBestMatchedCut() const;

  /// Finalize the cut set by removing duplicates and selecting the best
  /// pattern.
  ///
  /// This method:
  /// 1. Removes duplicate cuts based on inputs and root operation
  /// 2. Computes truth tables for surviving cuts (lazy computation)
  /// 3. Limits the number of cuts to prevent exponential growth
  /// 4. Matches each cut against available patterns
  /// 5. Selects the best pattern based on the optimization strategy
  void finalize(
      const CutRewriterOptions &options,
      llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut,
      llvm::BumpPtrAllocator &allocator, const LogicNetwork &logicNetwork);

  /// Get the number of cuts in this set.
  unsigned size() const;

  /// Add a new cut to this set using bump allocator.
  /// NOTE: The cut set must not be frozen
  void addCut(Cut *cut);

  /// Get read-only access to all cuts in this set.
  ArrayRef<Cut *> getCuts() const;
};

/// Configuration options for the cut-based rewriting algorithm.
///
/// These options control various aspects of the rewriting process including
/// optimization strategy, resource limits, and algorithmic parameters.
struct CutRewriterOptions {
  /// Optimization strategy (area vs. timing).
  OptimizationStrategy strategy;

  /// Maximum number of inputs allowed for any cut.
  /// Larger cuts provide more optimization opportunities but increase
  /// computational complexity exponentially.
  unsigned maxCutInputSize;

  /// Maximum number of cuts to maintain per logic node.
  /// The priority cuts algorithm keeps only the most promising cuts
  /// to prevent exponential explosion.
  unsigned maxCutSizePerRoot;

  /// Fail if there is a root operation that has no matching pattern.
  bool allowNoMatch = false;

  /// Put arrival times to rewritten operations.
  bool attachDebugTiming = false;

  /// Run priority cuts enumeration and dump the cut sets.
  bool testPriorityCuts = false;
};

//===----------------------------------------------------------------------===//
// Cut Enumeration Engine
//===----------------------------------------------------------------------===//

/// Cut enumeration engine for combinational logic networks.
///
/// The CutEnumerator is responsible for generating cuts for each node in a
/// combinational logic network. It uses a priority cuts algorithm to maintain a
/// bounded set of promising cuts while avoiding exponential explosion.
///
/// Implementation details are hidden via the pimpl idiom.
class CutEnumerator {
public:
  /// Constructor for cut enumerator.
  explicit CutEnumerator(const CutRewriterOptions &options);

  /// Enumerate cuts for all nodes in the given module.
  LogicalResult enumerateCuts(
      Operation *topOp,
      llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut =
          [](const Cut &) { return std::nullopt; });

  /// Create a new cut set for an index.
  CutSet *createNewCutSet(uint32_t index);

  /// Get the cut set for a specific index.
  const CutSet *getCutSet(uint32_t index);

  /// Clear all cut sets and return them as a vector.
  llvm::SmallVector<std::pair<uint32_t, CutSet *>> takeCutSets();

  /// Clear all cut sets and reset the enumerator.
  void clear();

  /// Dump cut information for debugging.
  void dump() const;

  /// Get cut sets (indexed by LogicNetwork index).
  const llvm::DenseMap<uint32_t, CutSet *> &getCutSets() const {
    return cutSets;
  }

  /// Get the processing order (indices in topological order).
  ArrayRef<uint32_t> getProcessingOrder() const { return processingOrder; }

private:
  /// Visit a single operation and generate cuts for it.
  LogicalResult visit(Operation *op);

  /// Visit a combinational logic operation and generate cuts.
  LogicalResult visitLogicOp(uint32_t nodeIndex);

  /// Maps indices to their associated cut sets.
  llvm::DenseMap<uint32_t, CutSet *> cutSets;

  /// Bump allocator for fast allocation of CutSet objects.
  llvm::BumpPtrAllocator allocator;

  /// Indices in processing order (topological).
  llvm::SmallVector<uint32_t> processingOrder;

  /// Configuration options for cut enumeration.
  const CutRewriterOptions &options;

  /// Function to match cuts against available patterns.
  llvm::function_ref<std::optional<MatchedPattern>(const Cut &)> matchCut;

  /// Flat logic network representation for efficient simulation.
  /// LogicNetwork is defined in the cpp file - opaque here for encapsulation.
  detail::LogicNetwork logicNetwork;
};

/// Base class for cut rewriting patterns used in combinational logic
/// optimization.
///
/// A CutRewritePattern represents a library component or optimization pattern
/// that can replace cuts in the combinational logic network. Each pattern
/// defines:
/// - How to recognize matching cuts and compute area/delay metrics
/// - How to transform/replace the matched cuts
///
/// Patterns can use truth table matching for efficient recognition or
/// implement custom matching logic for more complex cases.
struct CutRewritePattern {
  CutRewritePattern(mlir::MLIRContext *context) : context(context) {}
  /// Virtual destructor for base class.
  virtual ~CutRewritePattern() = default;

  /// Check if a cut matches this pattern and compute area/delay metrics.
  ///
  /// This method is called to determine if a cut can be replaced by this
  /// pattern. If the cut matches, it should return a MatchResult containing
  /// the area and per-input delays for this specific cut.
  ///
  /// If useTruthTableMatcher() returns true, this method is only
  /// called for cuts with matching truth tables.
  virtual std::optional<MatchResult> match(CutEnumerator &enumerator,
                                           const Cut &cut) const = 0;

  /// Specify truth tables that this pattern can match.
  ///
  /// If this method returns true, the pattern matcher will use truth table
  /// comparison for efficient pre-filtering. Only cuts with matching truth
  /// tables will be passed to the match() method. If it returns false, the
  /// pattern will be checked against all cuts regardless of their truth tables.
  /// This is useful for patterns that match regardless of their truth tables,
  /// such as LUT-based patterns.
  virtual bool
  useTruthTableMatcher(SmallVectorImpl<NPNClass> &matchingNPNClasses) const;

  /// Return a new operation that replaces the matched cut.
  ///
  /// Unlike MLIR's RewritePattern framework which allows arbitrary in-place
  /// modifications, this method creates a new operation to replace the matched
  /// cut rather than modifying existing operations. This constraint exists
  /// because the cut enumerator maintains references to operations throughout
  /// the circuit, making it safe to only replace the root operation of each
  /// cut while preserving all other operations unchanged.
  virtual FailureOr<Operation *> rewrite(mlir::OpBuilder &builder,
                                         CutEnumerator &enumerator,
                                         const Cut &cut) const = 0;

  /// Get the number of outputs this pattern produces.
  virtual unsigned getNumOutputs() const = 0;

  /// Get the name of this pattern. Used for debugging.
  virtual StringRef getPatternName() const { return "<unnamed>"; }

  /// Get location for this pattern(optional).
  virtual LocationAttr getLoc() const { return mlir::UnknownLoc::get(context); }

  mlir::MLIRContext *getContext() const { return context; }

private:
  mlir::MLIRContext *context;
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
class CutRewritePatternSet {
public:
  /// Constructor that takes ownership of the provided patterns.
  ///
  /// During construction, patterns are analyzed and organized for efficient
  /// lookup. Truth table matchers are indexed by their NPN canonical forms.
  CutRewritePatternSet(
      llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns);

private:
  /// Owned collection of all rewriting patterns.
  llvm::SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;

  /// Fast lookup table mapping NPN canonical forms to matching patterns.
  /// Each entry maps a truth table and input size to patterns that can handle
  /// it.
  DenseMap<std::pair<APInt, unsigned>,
           SmallVector<std::pair<NPNClass, const CutRewritePattern *>>>
      npnToPatternMap;

  /// Patterns that use custom matching logic instead of NPN lookup.
  /// These patterns are checked against every cut.
  SmallVector<const CutRewritePattern *, 4> nonNPNPatterns;

  /// CutRewriter needs access to internal data structures for pattern matching.
  friend class CutRewriter;
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
/// options.strategy = OptimizationStrategy::Area;
/// options.maxCutInputSize = 4;
/// options.maxCutSizePerRoot = 8;
///
/// CutRewritePatternSet patterns(std::move(optimizationPatterns));
/// CutRewriter rewriter(module, options, patterns);
///
/// if (failed(rewriter.run())) {
///   // Handle rewriting failure
/// }
/// ```
class CutRewriter {
public:
  /// Constructor for the cut rewriter.
  CutRewriter(const CutRewriterOptions &options, CutRewritePatternSet &patterns)
      : options(options), patterns(patterns), cutEnumerator(options) {}

  /// Execute the complete cut-based rewriting algorithm.
  ///
  /// This method orchestrates the entire rewriting process:
  /// 1. Enumerate cuts for all nodes in the combinational logic
  /// 2. Match cuts against available patterns
  /// 3. Select optimal patterns based on strategy
  /// 4. Rewrite the circuit with selected patterns
  LogicalResult run(Operation *topOp);

private:
  /// Enumerate cuts for all nodes in the given module.
  /// Note: This preserves module boundaries and does not perform
  /// rewriting across the hierarchy.
  LogicalResult enumerateCuts(Operation *topOp);

  /// Find patterns that match a cut's truth table.
  ArrayRef<std::pair<NPNClass, const CutRewritePattern *>>
  getMatchingPatternsFromTruthTable(const Cut &cut) const;

  /// Match a cut against available patterns and compute arrival time.
  std::optional<MatchedPattern> patternMatchCut(const Cut &cut);

  /// Perform the actual circuit rewriting using selected patterns.
  LogicalResult runBottomUpRewrite(Operation *topOp);

  /// Configuration options
  const CutRewriterOptions &options;

  /// Available rewriting patterns
  const CutRewritePatternSet &patterns;

  CutEnumerator cutEnumerator;
};

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_CUT_REWRITER_H
