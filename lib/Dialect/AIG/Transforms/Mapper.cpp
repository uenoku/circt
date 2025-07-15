//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs technology mapping on AIG circuits using cut-based
// algorithms. It implements priority cuts and supports both area-oriented
// and delay-oriented mapping modes.
//
// References:
// * A. Mishchenko, S. Cho, S. Chatterjee, and R. Brayton, "Combinational and
// sequential mapping with priority cuts", Proc. ICCAD '07
// * S. Chatterjee, A. Mishchenko, R. Brayton, X. Wang, and T. Kam, "Reducing
// structural bias in technology mapping", Proc. ICCAD'05
// * Fast Boolean Matching Based on NPN Classification, FPT 2013
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "aig-mapper"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_MAPPER
#define GEN_PASS_DEF_GENERICLUTMAPPER
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//
// TODO: Add power estimation
enum MappingStrategy { Area, Timing };

static bool compareDelayAndArea(MappingStrategy strategy, double lhsArea,
                                double lhsDelay, double rhsArea,
                                double rhsDelay) {
  if (MappingStrategy::Area == strategy) {
    // Compare by area only
    return lhsArea < rhsArea || (lhsArea == rhsArea && lhsDelay < rhsDelay);
  }
  if (MappingStrategy::Timing == strategy) {
    // Compare by delay only
    return lhsDelay < rhsDelay || (lhsDelay == rhsDelay && lhsArea < rhsArea);
  }
  llvm_unreachable("Unknown mapping strategy");
}

namespace {

struct TruthTable {
  unsigned numInputs;  // Number of inputs for this cut
  unsigned numOutputs; // Number of outputs for this cut
  APInt table;         // Truth table represented as an APInt
  TruthTable() = default;
  TruthTable(unsigned numInputs, unsigned numOutputs, const APInt &eval)
      : numInputs(numInputs), numOutputs(numOutputs), table(eval) {}
  TruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1u << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination
  APInt getOutput(const APInt &input) const {
    assert(input.getBitWidth() == numInputs && "Input width mismatch");
    return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
  }

  /// Set the output value for a given input combination
  void setOutput(const APInt &input, const APInt &output) {
    assert(input.getBitWidth() == numInputs && "Input width mismatch");
    assert(output.getBitWidth() == numOutputs && "Output width mismatch");
    unsigned offset = input.getZExtValue() * numOutputs;
    for (unsigned i = 0; i < numOutputs; ++i)
      table.setBitVal(offset + i, output[i]);
  }

  /// Apply input permutation to the truth table
  TruthTable
  applyPermutation(const SmallVectorImpl<unsigned> &permutation) const {
    assert(permutation.size() == numInputs && "Permutation size mismatch");
    TruthTable result(numInputs, numOutputs);

    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      APInt input(numInputs, i);
      APInt permutedInput(numInputs, 0);

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
      APInt input(numInputs, i);
      APInt negatedInput(numInputs, 0);

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
      APInt input(numInputs, i);
      APInt output = getOutput(input);
      APInt negatedOutput(numOutputs, 0);

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
  SmallVector<unsigned> inputPermutation;
  unsigned inputNegation = 0;
  unsigned outputNegation = 0;
  NPNClass() = default;
  NPNClass(const TruthTable &tt) : truthTable(tt) {}

  /// Compute "exact" NPN canonical form using the algorithm from the paper
  /// "Fast Boolean Matching Based on NPN Classification".
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
      SmallVector<unsigned> permutation(tt.numInputs);
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

  // Get semi canonical form for this NPN class
  static NPNClass computeSemiNPNCanonicalForm(const TruthTable &tt) {
    NPNClass semiCanonical(tt);
    size_t bitWidth = tt.table.getBitWidth();

    // Step 1. Determine the polarity of the output bits.
    // Check output polarity.
    if (2 * tt.table.countl_one() > bitWidth) {
      // If more than half of the bits are 1, we can flip the output polarity
      semiCanonical.outputNegation =
          (1 << tt.numOutputs) - 1; // All bits negated
      // Flip the truth table
      semiCanonical.truthTable.table.flipAllBits();
    } else {
      semiCanonical.outputNegation = 0;
    }

    // Step 2. Determine the polarity of inputs.

    semiCanonical.truthTable;

    // Step 3. Determine the input permutation.
    semiCanonical.inputPermutation.resize(tt.numInputs);
    for (unsigned i = 0; i < tt.numInputs; ++i)
      semiCanonical.inputPermutation[i] = i;

    // For semi-canonical form, we only consider input negation
    semiCanonical.inputNegation = 0;
    semiCanonical.outputNegation = 0;

    // The truth table is already in its canonical form
    semiCanonical.truthTable = tt;

    return semiCanonical;
  }
};

} // end anonymous namespace

static bool isAlwaysCutInput(Value value) {
  auto *op = value.getDefiningOp();
  // If the value has no defining operation, it is a primary input
  if (!op)
    return true;

  if (op->hasTrait<OpTrait::ConstantLike>()) {
    // Constant values are never cut inputs.
    return false;
  }

  // TODO: Extend this to allow comb.and/xor/or as well.
  return !isa<aig::AndInverterOp>(op);
}

class CutRewriterPatternSet;
class CutRewriter;
class CutRewriterPattern;

/// Represents a cut in the AIG network
struct Cut {
  llvm::SmallSetVector<Value, 4> inputs; // Cut inputs.
  // Operations in the cut. Operation is topologcally sorted, with the root
  // operation at the front.
  llvm::SmallSetVector<Operation *, 4> operations;

  bool isPrimaryInput() const {
    // A cut is a primary input if it has no operations and only one input
    return operations.empty() && inputs.size() == 1;
  }

  // If empty, the arrival time is not computed yet. If failure, there is no
  // matching pattern for this cut. If arrival time is success, it contains the
  // arrival time of the cut.
  std::optional<FailureOr<double>> arrivalTime; // Arrival time of the cut
  std::optional<CutRewriterPattern *> matchedPattern;

  Cut() = default;

  Operation *getRoot() const {
    return operations.empty()
               ? nullptr
               : operations.back(); // The first operation is the root
  }

  auto getArrivalTime() const { return arrivalTime; }
  auto getMatchedPattern() const { return *matchedPattern; }
  std::optional<FailureOr<double>> getArea() const;
  void setNoMatchingPattern() {
    this->matchedPattern.emplace(nullptr);
    this->arrivalTime = failure();
  }

  void setMatchedPatternAndArrivalTime(CutRewriterPattern *matchedPattern,
                                       double arrivalTime) {
    this->matchedPattern = matchedPattern;
    this->arrivalTime = arrivalTime;
  }

  // Cache for the truth table of this cut
  mutable std::optional<TruthTable> truthTable;
  // Compute the NPN canonical form for this cut
  mutable std::optional<NPNClass> computedNPN;

  const NPNClass &computeNPN() const {
    // If the truth table is already computed, return it
    if (computedNPN)
      return *computedNPN;

    // Compute the NPN (Negation Permutation Negation) canonical form for this
    // cut
    auto truthTable = getTruthTable();
    assert(succeeded(truthTable) && "Failed to compute truth table");

    // Compute the NPN canonical form
    auto canonicalForm = NPNClass::computeNPNCanonicalForm(*truthTable);

    computedNPN.emplace(std::move(canonicalForm));
    return *computedNPN;
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
    auto npnClass = computeNPN();
    llvm::dbgs() << npnClass.truthTable.table << "\n";

    llvm::dbgs() << "==========================\n";
  }

public:
  static Cut getAsPrimaryInput(Value value) {
    // Create a cut with a single root operation
    Cut cut;
    // There is no input for the primary input cut.
    cut.inputs.insert(value);

    return cut;
  }

  static Cut getSingletonCut(Operation *op) {
    // Create a cut with a single input value
    Cut cut;
    cut.operations.insert(op);
    for (auto value : op->getOperands()) {
      // Only consider integer values. No integer type such as seq.clock,
      // aggregate are pass through the cut.
      assert(value.getType().isInteger(1));

      cut.inputs.insert(value);
    }
    return cut;
  }

  Cut(Operation *root) {
    // Create a cut with the root operation
    operations.insert(root);
    for (auto value : root->getOperands()) {
      if (inputs.count(value) || !value.getType().isInteger(1))
        continue;

      auto *defOp = value.getDefiningOp();
      assert(defOp && "Value must have a defining operation");
      if (defOp) {
        operations.insert(defOp);
      }
    }
  }

  /// Merge this cut with another cut
  Cut mergeWith(const Cut &other, Operation *root) const {
    // Create a new cut that combines this cut and the other cut
    Cut newCut;
    SmallVector<Operation *, 4> worklist{root};

    // Topological sort the operations in the new cut.
    std::function<void(Operation *)> populateOperations = [&](Operation *op) {
      // If the operation is already in the cut, skip it
      if (newCut.operations.contains(op))
        return;

      // Add its operands to the worklist
      for (auto value : op->getOperands()) {
        if (isAlwaysCutInput(value)) {
          // If the value is a primary input, add it to the cut inputs
          continue;
        }

        // If the value is in *both* cuts inputs, it is an input. So skip it.
        bool isInput = inputs.contains(value);
        bool isOtherInput = other.inputs.contains(value);
        // If the value is in this cut inputs, it is an input. So skip it
        if (isInput && isOtherInput) {
          continue;
        }
        auto *defOp = value.getDefiningOp();

        assert(defOp);

        if (isInput) {
          if (!other.operations.contains(defOp)) // op is in the other cut.
            continue;
        }

        if (isOtherInput) {
          if (!operations.contains(defOp)) // op is in this cut.
            continue;
        }

        populateOperations(defOp);
      }

      // Add the operation to the cut
      newCut.operations.insert(op);
    };
    populateOperations(root);

    // Construct inputs.
    for (auto *operation : newCut.operations) {
      for (auto value : operation->getOperands()) {
        if (isAlwaysCutInput(value)) {
          newCut.inputs.insert(value);
          continue;
        }

        auto *defOp = value.getDefiningOp();
        assert(defOp && "Value must have a defining operation");
        // If the operation is not in the cut, it is an input
        if (!newCut.operations.contains(defOp))
          // Add the input to the cut
          newCut.inputs.insert(value);
      }
    }

    // Update area and delay based on the merged cuts
    return newCut;
  }

  /// Get the size of this cut
  unsigned getInputSize() const { return inputs.size(); }
  unsigned getCutSize() const { return operations.size(); }
  size_t getOutputSize() const { return getRoot()->getNumResults(); }

  FailureOr<TruthTable> getTruthTable() const {
    assert(!isPrimaryInput() && "Primary input cuts do not have truth tables");

    if (truthTable)
      return *truthTable;

    int64_t numInputs = getInputSize();
    int64_t numOutputs = getOutputSize();
    assert(numInputs < 20 && "Truth table is too large");
    // Simulate the IR.
    // For each input combination, compute the output values.

    // 2. Lower the cut to a LUT. We can get a truth table by evaluating the
    // cut body with every possible combination of the input values.
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
        return failure();
      }
      // Set the output value for the truth table
      for (size_t j = 0; j < op->getNumResults(); ++j) {
        auto outputValue = op->getResult(j);
        if (!outputValue.getType().isInteger(1)) {
          op->emitError("Output value is not a single bit: ") << outputValue;
          return failure();
        }
      }
    }

    // Extract the truth table from the root operation
    auto rootResults = getRoot()->getResults();
    assert(rootResults.size() == 1 &&
           "For now we only support single output cuts");
    auto result = rootResults[0];

    // Cache the truth table
    truthTable.emplace(numInputs, numOutputs, eval[result]);
    return *truthTable;
  }

  LogicalResult simulateOp(Operation *op,
                           DenseMap<Value, APInt> &values) const {
    if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
      auto inputs = andOp.getInputs();
      SmallVector<APInt, 2> operands;
      for (auto input : inputs) {
        auto it = values.find(input);
        if (it == values.end()) {
          op->emitError("Input value not found: ") << input;
          return failure();
        }
        operands.push_back(it->second);
      }

      // Simulate the AND operation
      values[andOp.getResult()] = andOp.evaluate(operands);
      return success();
    }
    // Add more operation types as needed
    return failure();
  }
};

class MatchedPattern {
public:
  CutRewriterPattern *pattern = nullptr; // Matched pattern
  Cut *cut = nullptr;                    // Cut that matched the pattern
  double arrivalTime;                    // Arrival time of the cut

  MatchedPattern() = default;
  MatchedPattern(CutRewriterPattern *pattern, Cut *cut, double arrivalTime)
      : pattern(pattern), cut(cut), arrivalTime(arrivalTime) {}

  bool isValid() const { return pattern != nullptr; }
};

/// Cut set for a node using priority cuts algorithm
class CutSet {
private:
  llvm::SmallVector<Cut, 12> cuts;
  std::optional<FailureOr<double>> arrivalTime;

public:
  virtual ~CutSet() = default;
  auto getArrivalTime() const { return arrivalTime; }
  Cut *getMatched() {
    if (cuts.empty())
      return nullptr;
    return &cuts[0];
  }

  // Keep the cuts sorted by priority. The priority is determined by the
  // optimization mode (area or timing). The cuts are sorted in descending
  // order. This also removes duplicate cuts based on their inputs and root
  // operation.
  void freezeCutSet(MappingStrategy storategy, unsigned maxCuts,
                    llvm::function_ref<bool(const Cut &, const Cut &)> compare,
                    llvm::function_ref<bool(Cut &)> hasMatching) {
    DenseSet<std::pair<ArrayRef<Value>, Operation *>> uniqueCuts;
    size_t uniqueCount = 0;
    for (size_t i = 0; i < cuts.size(); ++i) {
      auto &cut = cuts[i];
      // Create a unique identifier for the cut based on its inputs and root
      auto inputs = cut.inputs.getArrayRef();
      if (!uniqueCuts.contains({inputs, cut.getRoot()})) {
        if (i != uniqueCount) {
          // Move the unique cut to the front of the vector
          // This maintains the order of cuts while removing duplicates
          // by swapping with the last unique cut found.
          std::swap(cuts[uniqueCount], cuts[i]);
        }

        // Beaware of lifetime of arrayref.
        uniqueCuts.insert({cuts[uniqueCount].inputs.getArrayRef(),
                           cuts[uniqueCount].getRoot()});
        ++uniqueCount;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Original cuts: " << cuts.size()
                            << " Unique cuts: " << uniqueCount << "\n");
    // Resize the cuts vector to the number of unique cuts found
    cuts.resize(uniqueCount);

    // Maintain size limit by removing worst cuts
    if (cuts.size() > maxCuts) {
      // Sort by priority using the provided comparison function
      std::sort(cuts.begin(), cuts.end(), compare);

      // We should keep at least one cut that matches the current pattern
      // sets.
      bool matched = false;
      for (size_t i = 0; i < maxCuts; ++i) {
        if (hasMatching(cuts[i])) {
          matched = true;
        }
      }

      if (!matched) {
        // Ok, try searching a cut.
        for (size_t i = maxCuts; i < cuts.size(); ++i) {
          if (hasMatching(cuts[i])) {
            // Found a cut that matches the current pattern sets.
            std::swap(cuts[maxCuts - 1], cuts[i]);
            break;
          }
        }
      }

      cuts.resize(maxCuts);
    }

    // Find the best matching pattern for this cut set
    arrivalTime = std::numeric_limits<double>::max();
    double lhsArea = std::numeric_limits<double>::max();
    for (size_t i = 0; i < cuts.size(); ++i) {
      if (hasMatching(cuts[i])) {
        auto rhsTime = **cuts[i].getArrivalTime();
        auto rhsArea = **cuts[i].getArea();

        if (compareDelayAndArea(storategy, rhsArea, rhsTime, lhsArea,
                                **arrivalTime)) {
          arrivalTime = rhsTime;
          lhsArea = rhsArea;
          if (i)
            std::swap(cuts[0], cuts[i]);
        }
      }
    }
  }

  size_t size() const { return cuts.size(); }

  /// Add a cut to the set, maintaining priority order
  void addCut(Cut cut) { cuts.push_back(std::move(cut)); }

  /// Get the best cut based on optimization mode
  const Cut *
  getBestCut(llvm::function_ref<bool(const Cut &, const Cut &)> compare) const {
    if (cuts.empty())
      return nullptr;

    const Cut *best = &cuts[0];
    for (const Cut &cut : cuts) {
      if (compare(cut, *best))
        best = &cut; // Found a better cut
    }
    return best;
  }

  /// Iterator that returns all the cuts.
  /// This is a range-based for loop compatible iterator.
  ArrayRef<Cut> getCuts() const { return cuts; }

  virtual void notifyBestMatch(Cut &cut) {
    // This method is called when a cut matches a pattern.
    // It can be used to update the cut set or perform additional actions.
    // Default implementation does nothing.
  }

  /// Notify that a cut has failed to match a pattern.
  virtual void notifyMatchFailure(Cut &cut) {}

  /// Clear all cuts
  void clear() { cuts.clear(); }
};

struct CutSetWithAreaFlow : CutSet {
  double areaFlow = 0.0;
  // The node state is used to store the cut set for each node in the AIG
  // network. It is used to keep track of the cuts for each node and their
  virtual ~CutSetWithAreaFlow() = default;

  void notifyBestMatch(Cut &cut) override {}
  void notifyMatchFailure(Cut &cut) override {}
};

// Base class for tech libraries.
struct CutRewriterPattern {
  virtual ~CutRewriterPattern() = default;

  // Run heavy initialization logic here if needed.
  virtual LogicalResult initialize() { return success(); }

  // Return true if the cut matches this cut rewriter pattern. If
  // `useTruthTableMatcher` is implemented to return true, this method is
  // called only when the truth table matches the cut set.
  virtual bool match(const Cut &cut) const = 0;

  // If necessary, populate `truthTable` with the truth table this pattern
  // potentially matches.  and return true if the pattern is applicable if the
  // truth table matches the cut set.
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

  // Interface for cut priorities.
  virtual double getArea(const Cut &cut) const = 0;

  // Get delay for a specific input-output pair.
  virtual double getDelay(const Cut &cut, size_t inputIndex,
                          size_t outputIndex) const = 0;

  // TODO: Include power.

  virtual unsigned getNumInputs() const = 0;
  virtual unsigned getNumOutputs() const = 0;
};

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewriterPattern {
  hw::HWModuleOp module;

  TechLibraryPattern(hw::HWModuleOp mod) : module(mod) {}

  /// Match the cut set against this library primitive
  bool match(const Cut &cutSet) const override { return false; }

  /// Rewrite the cut set using this library primitive
  LogicalResult rewrite(mlir::PatternRewriter &rewriter,
                        Cut &cut) const override {
    // Compute permutations of inputs and outputs
    // and rewrite the cut using the library primitive.

    return success();
  }

  double getAttr(StringRef name) const {
    auto dict = module->getAttrOfType<DictionaryAttr>("hw.techlib.info");
    if (!dict)
      return 0.0; // No attributes available
    auto attr = dict.get(name);
    if (!attr)
      return 0.0; // Attribute not found
    return cast<FloatAttr>(attr).getValue().convertToDouble();
  }

  double getArea(const Cut &cut) const override { return getAttr("area"); }

  double getDelay(const Cut &cut, size_t inputIndex,
                  size_t outputIndex) const override {
    return getAttr("delay");
  }

  unsigned getNumInputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumInputPorts();
  }

  unsigned getNumOutputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumOutputPorts();
  }
};

struct TestTechLibraryPattern : public CutRewriterPattern {
  /// Test pattern for a simple AND gate
  bool match(const Cut &cutSet) const override {
    // Match if the cut has exactly 2 inputs and 1 output
    return cutSet.getInputSize() == 2 && cutSet.getOutputSize() == 1;
  }

  LogicalResult rewrite(mlir::PatternRewriter &rewriter,
                        Cut &cut) const override {
    // Create an AND operation with the inputs of the cut
    auto *root = cut.getRoot();
    rewriter.replaceOpWithNewOp<aig::AndInverterOp>(root, cut.inputs[0],
                                                    cut.inputs[1]);
    return success();
  }

  double getArea(const Cut &cut) const override { return 1.0; }
  double getDelay(const Cut &cut, size_t inputIndex,
                  size_t outputIndex) const override {
    return 1.0; // Fixed delay for AND gate
  }

  unsigned getNumInputs() const override { return 2; }
  unsigned getNumOutputs() const override { return 1; }
};

struct GenericLUT : public CutRewriterPattern {
  /// Generic LUT primitive with k inputs
  size_t k; // Number of inputs for the LUT
  GenericLUT(size_t k) : k(k) {}
  bool match(const Cut &cutSet) const override {
    // Check if the cut matches the LUT primitive
    LLVM_DEBUG(llvm::dbgs()
                   << "Matching cut set with " << cutSet.getInputSize()
                   << " inputs against LUT with " << k << " inputs.\n";);
    return cutSet.getInputSize() <= k;
  }

  unsigned getNumInputs() const override { return k; }
  unsigned getNumOutputs() const override { return 1; } // Single output LUT

  double getArea(const Cut &cut) const override {
    // TODO: Implement area-flow.
    return 1.0;
  }

  double getDelay(const Cut &cut, size_t inputIndex,
                  size_t outputIndex) const override {
    // Assume a fixed delay for the generic LUT
    return 1.0;
  }

  LogicalResult rewrite(mlir::PatternRewriter &rewriter,
                        Cut &cut) const override {
    // NOTE: Don't use NPN because it's necessary to consider polarity etc.
    auto truthTable = cut.getTruthTable();
    if (failed(truthTable))
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting cut with " << cut.getInputSize()
                   << " inputs and " << cut.getCutSize()
                   << " operations to a generic LUT with " << k << " inputs.\n";
      cut.dump();
      llvm::dbgs() << "Truth table: " << truthTable->table << "\n";
      for (size_t i = 0; i < truthTable->table.getBitWidth(); ++i) {
        for (size_t j = 0; j < cut.getInputSize(); ++j) {
          // Print the input values for the truth table
          llvm::dbgs() << (i & (1u << j) ? "1" : "0");
        }
        llvm::dbgs() << " " << (truthTable->table[i] ? "1" : "0") << "\n";
      }
    });

    SmallVector<bool> lutTable;
    // Convert the truth table to a LUT table
    for (uint32_t i = 0; i < truthTable->table.getBitWidth(); ++i)
      lutTable.push_back(truthTable->table[i]);

    auto arrayAttr = rewriter.getBoolArrayAttr(
        lutTable); // Create a boolean array attribute.

    SmallVector<Value> lutInputs(cut.inputs.rbegin(), cut.inputs.rend());

    // Generate comb.truth table operation.
    auto truthTableOp = rewriter.create<comb::TruthTableOp>(
        cut.getRoot()->getLoc(), lutInputs, arrayAttr);

    // Replace the root operation with the truth table operation
    rewriter.replaceOp(cut.getRoot(), truthTableOp);
    return success();
  }
};

/// Test function to verify NPN computation
static void testNPNComputation() {
  LLVM_DEBUG({
    // Create a simple 2-input truth table for AND function
    TruthTable tt(2, 1);

    // Set truth table for AND: (0,0)->0, (0,1)->0, (1,0)->0, (1,1)->1
    tt.setOutput(APInt(2, 0), APInt(1, 0)); // 00 -> 0
    tt.setOutput(APInt(2, 1), APInt(1, 0)); // 01 -> 0
    tt.setOutput(APInt(2, 2), APInt(1, 0)); // 10 -> 0
    tt.setOutput(APInt(2, 3), APInt(1, 1)); // 11 -> 1

    llvm::dbgs() << "Original truth table hash: " << tt.table.getZExtValue()
                 << "\n";

    // Test permutation
    SmallVector<unsigned> perm = {1, 0}; // Swap inputs
    TruthTable permuted = tt.applyPermutation(perm);
    llvm::dbgs() << "Permuted truth table hash: "
                 << permuted.table.getZExtValue() << "\n";

    // Test input negation
    unsigned negMask = 0b10; // Negate first input
    TruthTable negated = tt.applyInputNegation(negMask);
    llvm::dbgs() << "Negated truth table hash: " << negated.table.getZExtValue()
                 << "\n";

    // Test output negation
    unsigned outputNeg = 1; // Negate output
    TruthTable outputNegated = tt.applyOutputNegation(outputNeg);
    llvm::dbgs() << "Output negated truth table hash: "
                 << outputNegated.table.getZExtValue() << "\n";
  });
}

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

  // Maps NPN classes to patterns
  DenseMap<APInt, SmallVector<std::pair<NPNClass, CutRewriterPattern *>>>
      npnToPatternMap;
  SmallVector<CutRewriterPattern *, 4> nonTruthTablePatterns;

  friend class CutRewriter;
};

/// Main mapper class
class CutRewriter {
private:
  Operation *topOp;
  MappingStrategy strategy; // Mapping strategy (area or timing)
  unsigned maxCutInputSize;
  unsigned maxCutSizePerRoot;
  const CutRewriterPatternSet &patterns;
  llvm::MapVector<Value, std::unique_ptr<CutSet>> cutSets;
  std::function<bool(const Cut &, const Cut &)> compare;

public:
  CutRewriter(Operation *op, MappingStrategy strategy, unsigned maxCutInputSize,
              unsigned maxCutSizePerRoot, CutRewriterPatternSet &patterns)
      : topOp(op), strategy(strategy), maxCutInputSize(maxCutInputSize),
        maxCutSizePerRoot(maxCutSizePerRoot), patterns(patterns) {
    // Set the comparison function based on area or delay mode
    if (MappingStrategy::Area == strategy) {
      compare = [](const Cut &a, const Cut &b) {
        return a.getCutSize() < b.getCutSize();
      };
    } else if (MappingStrategy::Timing == strategy) {
      compare = [](const Cut &a, const Cut &b) {
        // Implement depth.
        assert(false);
        return true;
      };
    } // Initialize library
  }

  /// Initialize the technology library

  /// Run the mapping algorithm
  LogicalResult run() {
    LLVM_DEBUG(llvm::dbgs() << "Starting AIG technology mapping\n");
    LLVM_DEBUG(llvm::dbgs()
               << "Mode: "
               << (MappingStrategy::Area == strategy ? "area" : "delay")
               << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Max cut size: " << maxCutSizePerRoot << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "Max cuts per node: " << maxCutSizePerRoot << "\n");

    // Test NPN computation
    LLVM_DEBUG(testNPNComputation());

    // Process each HW module
    return mapModule(topOp);
  }

private:
  /// Map a single HW module
  LogicalResult mapModule(Operation *hwModule) {
    // Clear previous state
    cutSets.clear();

    // Step 1: Enumerate cuts for all nodes
    if (failed(enumerateCuts(hwModule)))
      return failure();

    // Step 2: Select best cuts and perform mapping
    if (failed(performMapping(hwModule)))
      return failure();

    return success();
  }

  /// Enumerate cuts for all nodes in the module
  LogicalResult enumerateCuts(Operation *hwModule) {
    LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

    // Topological traversal
    llvm::SmallVector<Operation *> worklist;
    llvm::DenseSet<Operation *> visited;

    auto result = hwModule->walk([&](Operation *op) {
      if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
        if (failed(generateCutsForAndOp(andOp)))
          return mlir::WalkResult::interrupt();
        auto &cutSet = getCutSet(andOp.getResult());
        LLVM_DEBUG(llvm::dbgs() << "Generated cut for AND operation: "
                                << andOp.getResult() << "\n");
        for (const Cut &cut : cutSet.getCuts()) {
          LLVM_DEBUG(cut.dump());
        }
      }

      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      hwModule->emitError("Failed to generate cuts for AND operations");
      return failure();
    }

    return success();
  }

  /// Get the cut set for a specific operation
  const CutSet &getCutSet(Value value) {
    // If the cut set does not exist, create a new one
    if (!cutSets.contains(value)) {
      // Add a trivial cut for primary inputs
      cutSets[value] = std::make_unique<CutSet>();
      cutSets[value]->addCut(Cut::getAsPrimaryInput(value));
    }

    return *cutSets.find(value)->second;
  }

  CutSet *getOrCreateCutSet(Value value) {
    // If the cut set does not exist, create a new one
    if (!cutSets.contains(value)) {
      // Add a trivial cut for primary inputs
      cutSets[value] = std::make_unique<CutSet>();
    }

    return cutSets.find(value)->second.get();
  }

  ArrayRef<std::pair<NPNClass, CutRewriterPattern *>>
  getMatchingPatternFromTruthTable(const Cut &cut) const {
    if (patterns.npnToPatternMap.empty())
      return {};

    auto &npnClass = cut.computeNPN();
    auto it = patterns.npnToPatternMap.find(npnClass.truthTable.table);
    if (it == patterns.npnToPatternMap.end())
      return {};
    return it->getSecond();
  }

  // Compute arrival time to this cut.
  void matchCutToPattern(Cut &cut) {
    if (cut.matchedPattern)
      return; // Cut already has a matched pattern
    if (cut.isPrimaryInput()) {
      cut.setMatchedPatternAndArrivalTime(nullptr, 0.0);
      return;
    }
    double bestArrivalTime = std::numeric_limits<double>::max();
    CutRewriterPattern *bestPattern = nullptr;
    SmallVector<double, 4> inputArrivalTimes, outputArrivalTimes;
    inputArrivalTimes.reserve(cut.getInputSize());
    outputArrivalTimes.reserve(cut.getOutputSize());
    for (auto input : cut.inputs) {
      assert(input.getType().isInteger(1));
      // If the input is a primary input, it has no delay
      auto *arrivalTime = cutSets.find(input);
      if (isAlwaysCutInput(input)) {
        // If the input is a primary input, it has no delay
        inputArrivalTimes.push_back(0.0);
      } else if (arrivalTime != cutSets.end()) {
        // If the arrival time is already computed, use it
        auto arrivalTimeOpt = arrivalTime->second->getArrivalTime();
        assert(arrivalTimeOpt &&
               "Cut must have a valid arrival time after pruning");
        if (failed(*arrivalTimeOpt)) {
          cut.setNoMatchingPattern();
          return;
        }

        inputArrivalTimes.push_back(**arrivalTimeOpt);
      } else {
        // If the input is not a primary input, it has no arrival time yet
        cut.setNoMatchingPattern();
        return;
      }
    }
    auto tryPattern = [&](CutRewriterPattern *pattern) {
      if (!pattern->match(cut))
        return;
      // If the pattern matches the cut, compute the arrival time
      double patternArrivalTime = 0.0;
      // Compute the maximum delay for each output from inputs
      for (size_t i = 0; i < cut.getInputSize(); ++i) {
        for (size_t j = 0; j < cut.getOutputSize(); ++j) {
          patternArrivalTime =
              std::max(patternArrivalTime,
                       pattern->getDelay(cut, i, j) + inputArrivalTimes[i]);
        }
      }

      if (!bestPattern ||
          compareDelayAndArea(strategy, pattern->getArea(cut),
                              patternArrivalTime, bestPattern->getArea(cut),
                              bestArrivalTime)) {
        bestArrivalTime = patternArrivalTime;
        bestPattern = pattern;
      }
    };

    for (auto &[_, pattern] : getMatchingPatternFromTruthTable(cut))
      tryPattern(pattern);

    for (CutRewriterPattern *pattern : patterns.nonTruthTablePatterns)
      tryPattern(pattern);

    // Set the arrival time for the cut
    if (bestPattern) {
      cut.setMatchedPatternAndArrivalTime(bestPattern, bestArrivalTime);
    } else {
      cut.setNoMatchingPattern();
    }
  }

  /// Generate cuts for AND-inverter operations
  LogicalResult generateCutsForAndOp(Operation *logicOp) {
    assert(logicOp->getNumResults() == 1 &&
           "Logic operation must have a single result");
    Value result = logicOp->getResult(0);

    size_t numOperands = logicOp->getNumOperands();
    if (numOperands > 2)
      return logicOp->emitError("when computing cuts operation must have 1 or "
                                "2 operands, found: ")
             << numOperands;

    if (!logicOp->getOpResult(0).getType().isInteger(1))
      return logicOp->emitError("Result type must be a single bit integer");

    Cut singletonCut = Cut::getSingletonCut(logicOp);

    auto *resultCutSet = getOrCreateCutSet(result);

    auto prune = llvm::make_scope_exit([&]() {
      // Prune the cut set to maintain the maximum number of cuts
      getOrCreateCutSet(result)->freezeCutSet(
          strategy, maxCutSizePerRoot, compare, [&](Cut &cut) {
            // Check if the cut matches any pattern
            matchCutToPattern(cut);
            return succeeded(*cut.getArrivalTime());
          });
      /*
            for (const Cut &cut : resultCutSet.getCuts()) {
              auto arrivalTime = cut.getArrivalTime();
              assert(arrivalTime &&
                     "Cut must have a valid arrival time after pruning");
              if (failed(*arrivalTime))
                continue;
              // Update the arrival time for the result
              auto matchedPattern = cut.getMatchedPattern();
              assert(matchedPattern &&
                     "Cut must have a matched pattern after pruning");
              auto [it, inserted] = arrivalTimes.try_emplace(result,
         **arrivalTime); if (!inserted) {
                // If the arrival time already exists, take the minimum
                it->second = std::min(it->second, **arrivalTime);
              }
            }
              */
    });

    // Operation itself is a cut, so add it to the cut set
    resultCutSet->addCut(singletonCut);

    if (numOperands == 1) {
      auto inputCutSet = getCutSet(logicOp->getOperand(0));
      for (const Cut &cut : inputCutSet.getCuts()) {
        // Merge the singleton cut with the input cut
        Cut newCut = cut.mergeWith(singletonCut, logicOp);
        if (newCut.getInputSize() > maxCutInputSize)
          continue; // Skip if merged cut exceeds max input size
        resultCutSet->addCut(std::move(newCut));
      }
      return success();
    }

    auto lhs = getCutSet(logicOp->getOperand(0));
    auto rhs = getCutSet(logicOp->getOperand(1));
    // Combine cuts from both inputs
    for (const Cut &cut0 : lhs.getCuts()) {
      for (const Cut &cut1 : rhs.getCuts()) {
        auto mergedCut = cut0.mergeWith(cut1, logicOp);
        if (mergedCut.getCutSize() > maxCutSizePerRoot)
          continue; // Skip cuts that are too large
        if (mergedCut.getInputSize() > maxCutInputSize)
          continue; // Skip if merged cut exceeds max input size
        resultCutSet->addCut(std::move(mergedCut));
      }
    }

    return success();
  }

  /// Perform the actual technology mapping
  LogicalResult performMapping(Operation *hwModule) {
    LLVM_DEBUG(llvm::dbgs() << "Performing technology mapping...\n");

    // For now, just report the cuts found
    unsigned totalCuts = 0;
    LLVM_DEBUG(llvm::dbgs() << "Total cuts enumerated: " << totalCuts << "\n");

    // TODO: Implement actual technology mapping transformation
    // This would involve:
    // 1. Select best cuts for each node
    // 2. Replace AIG nodes with library primitives
    // 3. Connect the mapped circuit
    auto cutVector = cutSets.takeVector();
    UnusedOpPruner pruner;
    PatternRewriter rewriter(hwModule->getContext());
    for (auto &[value, cutSet] : llvm::reverse(cutVector)) {
      if (value.use_empty()) {
        if (auto *op = value.getDefiningOp())
          pruner.eraseNow(op);
        continue;
      }

      if (isAlwaysCutInput(value)) {
        // If the value is a primary input, skip it
        LLVM_DEBUG(llvm::dbgs() << "Skipping primary input: " << value << "\n");
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Cut set for value: " << value << "\n");
      auto *bestCut = cutSet->getMatched();
      if (!bestCut) {
        return mlir::emitError(hwModule->getLoc(),
                               "No matching cut found for value: ")
               << value;
      }

      LLVM_DEBUG(llvm::dbgs() << "Best cut: " << bestCut->getRoot() << "\n");
      // Match the cut to a pattern and rewrite it
      auto *matchedPattern = bestCut->getMatchedPattern();
      if (!matchedPattern) {
        llvm::dbgs() << "Cut set size: " << cutSet->size() << "\n";
        // If no pattern matched, we cannot rewrite the cut
        bestCut->dump();
        continue;
      }
      assert(matchedPattern && "Cut must have a matched pattern after pruning");
      assert(matchedPattern->match(*bestCut) &&
             "Cut must match the pattern after pruning");

      rewriter.setInsertionPoint(bestCut->getRoot());
      if (failed(matchedPattern->rewrite(rewriter, *bestCut))) {
        return failure();
      }
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Mapper Pass
//===----------------------------------------------------------------------===//

namespace {
struct MapperPass : public impl::MapperBase<MapperPass> {
  using MapperBase<MapperPass>::MapperBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (libraryModules.empty())
      return;

    auto &symbolTable = getAnalysis<SymbolTable>();
    SmallVector<std::unique_ptr<CutRewriterPattern>> libraryPatterns;

    // Find library modules in the top module
    for (const std::string &moduleName : libraryModules) {
      // Find the module in the symbol table
      auto hwModule = symbolTable.lookup<hw::HWModuleOp>(moduleName);
      if (!hwModule) {
        module->emitError("Library module not found: ") << moduleName;
        signalPassFailure();
        return;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Found library module: " << moduleName << "\n");
      // Create a CutRewriterPattern for the library module
      std::unique_ptr<CutRewriterPattern> pattern =
          std::make_unique<TechLibraryPattern>(hwModule);

      // Initialize the pattern.
      // TODO: Consider initialize asynchronously.
      if (failed(pattern->initialize())) {
        module->emitError("Failed to initialize library pattern: ")
            << moduleName;
        signalPassFailure();
        return;
      }

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    CutRewriterPatternSet patternSet(std::move(libraryPatterns));
    CutRewriter mapper(module, MappingStrategy::Area, maxCutSize,
                       maxCutsPerNode, patternSet);

    if (failed(mapper.run()))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Generic LUT Mapper Pass
//===----------------------------------------------------------------------===//
struct GenericLUTMapperPass
    : public impl::GenericLutMapperBase<GenericLUTMapperPass> {
  using GenericLutMapperBase<GenericLUTMapperPass>::GenericLutMapperBase;
  void runOnOperation() override {

    // Add LUT pattern.
    auto module = getOperation();
    SmallVector<std::unique_ptr<CutRewriterPattern>> patterns;
    patterns.push_back(std::make_unique<GenericLUT>(maxLutSize));
    CutRewriterPatternSet patternSet(std::move(patterns));

    CutRewriter mapper(module, MappingStrategy::Area, maxLutSize,
                       maxCutsPerNode, patternSet);
    if (failed(mapper.run()))
      signalPassFailure();
  }
}; // namespace
} // namespace

std::optional<FailureOr<double>> Cut::getArea() const {
  if (isPrimaryInput())
    return 0.0; // Primary inputs do not have area
  if (!matchedPattern)
    return {}; // Matching is not performed yet.

  if (!*matchedPattern)
    return failure();

  return (*matchedPattern)->getArea(*this);
}