//===- Mapper.cpp - AIG Technology Mapping Pass ----------------*- C++ -*-===//
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
// The implementaion is based on following papers:
//
// * Combinational and Sequential Mapping with Priority Cuts, ICCAD 2007,
//   Alan M. et al.
// * Reducing Structural Bias in Technology Mapping, ICCAD 2006, Satrajit C. et
// al.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <functional>
#include <memory>

#define DEBUG_TYPE "aig-mapper"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_MAPPER
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Cut Data Structures
//===----------------------------------------------------------------------===//

namespace {

struct TruthTable {
  unsigned numInputs;  // Number of inputs for this cut
  unsigned numOutputs; // Number of outputs for this cut
  APInt table;         // Truth table represented as an APInt
  TruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1 << numInputs) * numOutputs, 0) {}

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
    for (unsigned i = 0; i < numOutputs; ++i) {
      table.setBitVal(offset + i, output[i]);
    }
  }

  /// Apply input permutation to the truth table
  TruthTable applyPermutation(const SmallVector<unsigned> &permutation) const {
    assert(permutation.size() == numInputs && "Permutation size mismatch");
    TruthTable result(numInputs, numOutputs);
    
    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      APInt input(numInputs, i);
      APInt permutedInput(numInputs, 0);
      
      // Apply permutation
      for (unsigned j = 0; j < numInputs; ++j) {
        permutedInput.setBitVal(j, input[permutation[j]]);
      }
      
      result.setOutput(permutedInput, getOutput(input));
    }
    
    return result;
  }

  /// Apply input negation to the truth table
  TruthTable applyInputNegation(const SmallVector<bool> &negation) const {
    assert(negation.size() == numInputs && "Negation size mismatch");
    TruthTable result(numInputs, numOutputs);
    
    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      APInt input(numInputs, i);
      APInt negatedInput(numInputs, 0);
      
      // Apply negation
      for (unsigned j = 0; j < numInputs; ++j) {
        negatedInput.setBitVal(j, negation[j] ? !input[j] : input[j]);
      }
      
      result.setOutput(negatedInput, getOutput(input));
    }
    
    return result;
  }

  /// Apply output negation to the truth table
  TruthTable applyOutputNegation(const SmallVector<bool> &negation) const {
    assert(negation.size() == numOutputs && "Negation size mismatch");
    TruthTable result(numInputs, numOutputs);
    
    for (unsigned i = 0; i < (1u << numInputs); ++i) {
      APInt input(numInputs, i);
      APInt output = getOutput(input);
      APInt negatedOutput(numOutputs, 0);
      
      // Apply negation
      for (unsigned j = 0; j < numOutputs; ++j) {
        negatedOutput.setBitVal(j, negation[j] ? !output[j] : output[j]);
      }
      
      result.setOutput(input, negatedOutput);
    }
    
    return result;
  }

  /// Get canonical hash for comparison
  uint64_t getHash() const {
    // Simple hash based on the table content
    uint64_t hash = 0;
    for (unsigned i = 0; i < table.getBitWidth(); ++i) {
      if (table[i]) {
        hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
    }
    return hash;
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

/// NPN canonical form representation
struct NPNCanonicalForm {
  TruthTable canonicalTT;
  SmallVector<unsigned> inputPermutation;
  SmallVector<bool> inputNegation;
  SmallVector<bool> outputNegation;
  
  NPNCanonicalForm(const TruthTable &tt) : canonicalTT(tt) {}
};

} // end anonymous namespace

/// Represents a cut in the AIG network
struct Cut {
  llvm::SmallSetVector<Value, 4> inputs; // Cut inputs (leaves)
  // Operations in the cut. Operation is topologcally sorted, with the root
  // operation at the front.
  llvm::SmallSetVector<Operation *, 4> operations;

  double delay = 0; // Delay of the cut
  double area = 0;  // Area of the cut

  Cut() = default;

  Operation *getRoot() const {
    return operations.front(); // The first operation is the root
  }  FailureOr<TruthTable> computeNPN() {
    // Compute the NPN (Negation Permutation Negation) canonical form for this cut
    auto truthTable = getTruthTable();
    if (failed(truthTable))
      return failure();

    // For single-output functions, compute NPN canonical form
    if (truthTable->numOutputs == 1) {
      auto canonicalForm = computeNPNCanonicalForm(*truthTable);
      return canonicalForm.canonicalTT;
    }

    // For multi-output functions, return the original truth table
    return *truthTable;
  }

private:
  /// Compute NPN canonical form using the algorithm from the paper
  NPNCanonicalForm computeNPNCanonicalForm(const TruthTable &tt) {
    NPNCanonicalForm canonical(tt);
    
    // Initialize permutation and negation vectors
    canonical.inputPermutation.resize(tt.numInputs);
    canonical.inputNegation.resize(tt.numInputs, false);
    canonical.outputNegation.resize(tt.numOutputs, false);
    
    // Initialize identity permutation
    for (unsigned i = 0; i < tt.numInputs; ++i) {
      canonical.inputPermutation[i] = i;
    }
    
    TruthTable bestTT = tt;
    
    // Try all possible input negations (2^n combinations)
    for (unsigned negMask = 0; negMask < (1u << tt.numInputs); ++negMask) {
      SmallVector<bool> inputNeg(tt.numInputs);
      for (unsigned i = 0; i < tt.numInputs; ++i) {
        inputNeg[i] = (negMask >> i) & 1;
      }
      
      TruthTable negatedTT = tt.applyInputNegation(inputNeg);
      
      // Try all possible permutations
      SmallVector<unsigned> permutation(tt.numInputs);
      for (unsigned i = 0; i < tt.numInputs; ++i) {
        permutation[i] = i;
      }
      
      do {
        TruthTable permutedTT = negatedTT.applyPermutation(permutation);
        
        // Try output negation (for single output)
        if (tt.numOutputs == 1) {
          SmallVector<bool> outputNeg(1, false);
          TruthTable candidate = permutedTT;
          
          // Try without output negation
          if (candidate.isLexicographicallySmaller(bestTT)) {
            bestTT = candidate;
            canonical.canonicalTT = candidate;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = inputNeg;
            canonical.outputNegation = outputNeg;
          }
          
          // Try with output negation
          outputNeg[0] = true;
          candidate = permutedTT.applyOutputNegation(outputNeg);
          if (candidate.isLexicographicallySmaller(bestTT)) {
            bestTT = candidate;
            canonical.canonicalTT = candidate;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = inputNeg;
            canonical.outputNegation = outputNeg;
          }
        } else {
          // For multi-output, just check without output negation
          if (permutedTT.isLexicographicallySmaller(bestTT)) {
            bestTT = permutedTT;
            canonical.canonicalTT = permutedTT;
            canonical.inputPermutation = permutation;
            canonical.inputNegation = inputNeg;
          }
        }
      } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
    
    return canonical;
  }

public:



  Cut(Operation *root) {
    // Create a cut with the root operation
    operations.insert(root);
    for (auto value : root->getOperands()) {
      // Only consider integer values. No integer type such as seq.clock,
      // aggregate are pass through the
      if (!value.getType().isInteger(1))
        continue;
      auto *defOp = value.getDefiningOp();
      if (defOp) {
        operations.insert(defOp);
      } else {
        // If the value has no defining operation, it is an input
        inputs.insert(value);
      }
    }
    // Update area and delay based on the root operation
    area = 0.0;  // TODO: Set area based on the root operation
    delay = 0.0; // TODO: Set delay based on the root operation
  }

  /// Merge this cut with another cut
  Cut mergeWith(const Cut &other, Operation *root) const {
    // Create a new cut that combines this cut and the other cut
    Cut newCut;
    SmallVector<Operation *, 4> worklist{root};
    while (!worklist.empty()) {
      auto *op = worklist.pop_back_val();
      auto result = newCut.operations.insert(op);
      if (result) {
        for (auto value : op->getOperands()) {
          // Only consider integer values. No integer type such as seq.clock,
          // aggregate are pass through the cut.
          if (!value.getType().isInteger(1))
            continue;

          if (inputs.contains(value) && other.inputs.contains(value)) {
            // If the value is in *both* cuts, it is an input. The input is
            // constructed later, so we skip it here.
            continue;
          }

          auto *defOp = value.getDefiningOp();
          if (defOp) {
            worklist.push_back(defOp);
          } else {
            // Block arguments or other values without defining operations
            assert(false && "Block arguments must have been added as inputs");
          }
        }
      }
    }

    // Construct inputs. Reverse the order to ensure inputs are added in the
    // consistent order
    for (auto *operation : llvm::reverse(newCut.operations)) {
      for (auto value : operation->getOperands()) {
        // No constnant values won't be added to the cut inputs
        if (!value.getType().isInteger(1))
          continue;
        bool isInput = false;
        if (auto *defOp = value.getDefiningOp()) {
          // If the operation is not in the cut, it is an input
          isInput = !newCut.operations.contains(defOp);
        } else {
          isInput = true;
        }

        if (isInput)
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

  bool isOutputSingleBit() const {
    // Check if the output is a single bit
    size_t count = 0;
    for (auto result : getRoot()->getResults()) {
      if (!result.getType().isInteger(1) || count++ > 0)
        return false; // Not a single bit output
    }
    return true; // All outputs are single bit
  }

  /// Check if this cut dominates another cut (better in all metrics)
  /// Priority cuts algorithm uses area and delay as metrics.
  bool dominates(const Cut &other) const {
    return area < other.area && delay < other.delay;
  }

  FailureOr<TruthTable> getTruthTable() {
    int64_t numInputs = getInputSize();
    int64_t numOutputs = getOutputSize();
    TruthTable tt(numInputs, numOutputs);
    assert(numInputs < 16 && "Too many inputs for truth table");
    // Simulate the IR.
    // For each input combination, compute the output values.

    // 2. Lower the cut to a LUT. We can get a truth table by evaluating the cut
    // body with every possible combination of the input values.
    uint32_t tableSize = 1 << numInputs;
    DenseMap<Value, APInt> eval;
    for (uint32_t i = 0; i < numInputs; ++i) {
      APInt value(tableSize, 0);
      for (uint32_t j = 0; j < tableSize; ++j) {
        // Make sure the order of the bits is correct.
        value.setBitVal(j, (j >> i) & 1);
      }
      // Set the input value for the truth table
      eval[inputs[i]] = value;
    }

    // Simulate the operations in the cut
    for (auto *op : operations) {
      auto result = simulateOp(op, eval);
      if (failed(result)) {
        llvm::errs() << "Failed to simulate operation: " << op->getName()
                     << "\n";
        return failure();
      }
      // Set the output value for the truth table
      for (size_t j = 0; j < op->getNumResults(); ++j) {
        auto outputValue = op->getResult(j);
        if (!outputValue.getType().isInteger(1)) {
          llvm::errs() << "Output value is not a single bit: " << outputValue
                       << "\n";
          return failure();
        }
      }
    }

    // Extract the truth table from the root operation
    auto rootResults = getRoot()->getResults();
    for (size_t i = 0; i < rootResults.size(); ++i) {
      auto result = rootResults[i];
      if (eval.find(result) != eval.end()) {
        APInt resultValue = eval[result];
        // Set the truth table entries
        for (uint32_t j = 0; j < tableSize; ++j) {
          APInt input(numInputs, j);
          APInt output(numOutputs, resultValue[j] ? 1 : 0);
          tt.setOutput(input, output);
        }
      }
    }

    return tt;
  }

  LogicalResult simulateOp(Operation *op, DenseMap<Value, APInt> &values) {
    if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
      auto inputs = andOp.getInputs();
      if (inputs.size() != 2) {
        return failure(); // Only support binary operations for now
      }
      
      auto lhs = values[inputs[0]];
      auto rhs = values[inputs[1]];
      APInt result = lhs & rhs;
      
      // Apply inversions
      if (andOp.isInverted(0)) {
        lhs = ~lhs;
        result = lhs & rhs;
      }
      if (andOp.isInverted(1)) {
        rhs = ~rhs;
        result = lhs & rhs;
      }
      
      values[andOp.getResult()] = result;
      return success();
    }
    // Add more operation types as needed
    return failure();
  }

  APInt simulate(const APInt &input) {
    DenseMap<Value, APInt> values;
    size_t bitPos = 0;
    for (auto value : inputs) {
      assert(value.getType().isInteger(1));
      values[value] = input.extractBits(1, bitPos++);
    }
    for (auto *op : operations) {
      if (failed(simulateOp(op, values))) {
        return APInt(1, 0);
      }
    }

    // The root operation should have the output value
    auto rootResults = getRoot()->getResults();
    if (!rootResults.empty()) {
      return values[rootResults[0]];
    }
    return APInt(1, 0);
  }
};

/// Cut set for a node using priority cuts algorithm
class CutSet {
private:
  llvm::SmallVector<Cut, 12> cuts;

public:
  // Keep the cuts sorted by priority. The priority is determined by the
  // optimization mode (area or timing). The cuts are sorted in descending
  // order,
  void pruneCut(unsigned maxCuts,
                llvm::function_ref<bool(const Cut &, const Cut &)> compare) {
    // Maintain size limit by removing worst cuts
    if (cuts.size() > maxCuts) {
      // Sort by priority using the provided comparison function
      std::sort(cuts.begin(), cuts.end(), compare);
      cuts.resize(maxCuts);
    }
  }

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

  /// Get all cuts
  ArrayRef<Cut> getCuts() const { return cuts; }

  /// Clear all cuts
  void clear() { cuts.clear(); }
};

// Base class for tech libraries.
struct CutRewriterPattern {
  virtual ~CutRewriterPattern() = default;

  // Run heavy initialization logic here if needed.
  virtual LogicalResult initialize() { return success(); }

  // Return true if the cut matches this library primitive. If
  // `useTruthTableMatcher` is implemented to return true, this method is
  // called only when the truth table matches the cut set.
  virtual bool match(const Cut &cutSet) const = 0;

  // Populate truthtable set and return true if the pattern is applicable if
  // the truth table matches the cut set.
  virtual bool useTruthTableMatcher(SmallVectorImpl<APInt> &truthTable) const {
    return false;
  }

  // Rewrite the cut. This is similar to MLIR's PatternRewriter. If success
  // is returned, the cut root operation must be replaced with other
  // operation.
  virtual LogicalResult rewrite(PatternRewriter &rewriter,
                                Cut &cutSet) const = 0;

  // Benefit of the cut in the library
  virtual double getArea() const = 0;

  // Get delay for a specific input-output pair.
  virtual double getDelay(size_t inputIndex, size_t outputIndex) const = 0;

  // TODO: Right now it's very simple. Power could be also very complicated
  // (usually it's based on internal states). So consider extending this to
  // support more complex power calculations.
  virtual double getPower() const = 0;

  virtual unsigned getNumInputs() const = 0;
  virtual unsigned getNumOutputs() const = 0;
};

/// Technology library primitive
struct LibraryPrimitive : public CutRewriterPattern {
  hw::HWModuleOp module;

  LibraryPrimitive(hw::HWModuleOp mod) : module(mod) {}

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

  double getArea() const override { return getAttr("area"); }

  double getDelay(size_t inputIndex, size_t outputIndex) const override {
    return getAttr("delay");
  }

  double getPower() const override { return getAttr("power"); }

  unsigned getNumInputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumInputPorts();
  }

  unsigned getNumOutputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumOutputPorts();
  }
};

struct GenericLUT : public CutRewriterPattern {
  /// Generic LUT primitive with k inputs
  size_t k; // Number of inputs for the LUT
  GenericLUT(size_t k) : k(k) {}
  bool match(const Cut &cutSet) const override {
    // Check if the cut matches the LUT primitive
    if (cutSet.getInputSize() > getNumInputs())
      return false;

    return cutSet.isOutputSingleBit();
  }

  double getArea() const override {
    // Assume a fixed area for the generic LUT
    return 1.0; // Placeholder value
  }
  double getDelay(size_t inputIndex, size_t outputIndex) const override {
    // Assume a fixed delay for the generic LUT
    return 1.0; // Placeholder value
  }
  double getPower() const override {
    // Assume a fixed power for the generic LUT
    return 1.0; // Placeholder value
  }

  LogicalResult rewrite(mlir::PatternRewriter &rewriter,
                        Cut &cut) const override {
    // TODO: Implement the actual rewrite logic
    auto truthTable = cut.getTruthTable();
    if (failed(truthTable))
      return failure();
    
    // Generate comb.truth table operation.
    // auto truthTableOp = rewriter.create<comb::TruthTableOp>(
    //     cut.getRoot()->getLoc(), truthTable->table, truthTable->numInputs,
    //     truthTable->numOutputs);

    // Replace the root operation with the truth table operation
    // rewriter.replaceOp(cut.getRoot(), truthTableOp);
    return success();
  }
};

// TODO: Add power estimation
enum MappingStrategy { Area, Timing };

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
    
    llvm::dbgs() << "Original truth table hash: " << tt.getHash() << "\n";
    
    // Test permutation
    SmallVector<unsigned> perm = {1, 0}; // Swap inputs
    TruthTable permuted = tt.applyPermutation(perm);
    llvm::dbgs() << "Permuted truth table hash: " << permuted.getHash() << "\n";
    
    // Test input negation
    SmallVector<bool> negation = {true, false}; // Negate first input
    TruthTable negated = tt.applyInputNegation(negation);
    llvm::dbgs() << "Negated truth table hash: " << negated.getHash() << "\n";
    
    // Test output negation
    SmallVector<bool> outputNeg = {true}; // Negate output
    TruthTable outputNegated = tt.applyOutputNegation(outputNeg);
    llvm::dbgs() << "Output negated truth table hash: " << outputNegated.getHash() << "\n";
  });
}

/// Main mapper class
class AIGMapper {
private:
  ModuleOp topModule;
  MappingStrategy strategy; // Mapping strategy (area or timing)
  unsigned maxCutSize;
  unsigned maxCutInputSize = 6; // Default max cut input size
  unsigned maxCutsPerNode;

  // Technology library
  llvm::SmallVector<std::unique_ptr<CutRewriterPattern>> library;
  llvm::DenseMap<Value, CutSet> cutSets;
  llvm::DenseMap<Value, double> arrivalTimes;
  std::function<bool(const Cut &, const Cut &)> compare;

public:
  AIGMapper(ModuleOp module, MappingStrategy strategy, unsigned maxCutSize,
            unsigned maxCutsPerNode, ArrayRef<std::string> libraryModules)
      : topModule(module), strategy(strategy), maxCutSize(maxCutSize),
        maxCutsPerNode(maxCutsPerNode) {
    // Set the comparison function based on area or delay mode
    if (MappingStrategy::Area == strategy) {
      compare = [](const Cut &a, const Cut &b) {
        return a.area < b.area || (a.area == b.area && a.delay < b.delay);
      };
    } else if (MappingStrategy::Timing == strategy) {
      compare = [](const Cut &a, const Cut &b) {
        return a.delay < b.delay || (a.delay == b.delay && a.area < b.area);
      };
    } // Initialize library
    initializeLibrary(libraryModules);
  }

  /// Initialize the technology library
  void initializeLibrary(ArrayRef<std::string> libraryModules) {
    if (libraryModules.empty())
      return;

    // Find library modules in the top module
    for (const std::string &moduleName : libraryModules) {
      topModule.walk([&](hw::HWModuleOp hwModule) {
        if (hwModule.getModuleName() == moduleName) {
          std::unique_ptr<CutRewriterPattern> it =
              std::unique_ptr<CutRewriterPattern>(
                  new LibraryPrimitive(hwModule));
          library.emplace_back(std::move(it));
          LLVM_DEBUG(llvm::dbgs() << "Added library module: " << moduleName);
        }
      });
    }
  }

  /// Run the mapping algorithm
  LogicalResult runMapper() {
    LLVM_DEBUG(llvm::dbgs() << "Starting AIG technology mapping\n");
    LLVM_DEBUG(llvm::dbgs()
               << "Mode: "
               << (MappingStrategy::Area == strategy ? "area" : "delay")
               << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Max cut size: " << maxCutSize << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Max cuts per node: " << maxCutsPerNode << "\n");

    // Test NPN computation
    LLVM_DEBUG(testNPNComputation());

    // Process each HW module
    for (hw::HWModuleOp hwModule : topModule.getOps<hw::HWModuleOp>()) {
      if (failed(mapModule(hwModule)))
        return failure();
    }

    return success();
  }

private:
  /// Map a single HW module
  LogicalResult mapModule(hw::HWModuleOp hwModule) {
    LLVM_DEBUG(llvm::dbgs()
               << "Mapping module: " << hwModule.getModuleName() << "\n");

    // Clear previous state
    cutSets.clear();
    arrivalTimes.clear();

    // Step 1: Enumerate cuts for all nodes
    if (failed(enumerateCuts(hwModule)))
      return failure();

    // Step 2: Select best cuts and perform mapping
    if (failed(performMapping(hwModule)))
      return failure();

    return success();
  }

  LogicalResult enumerateCuts(Operation *op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Enumerating cuts for operation: " << op->getName() << "\n");

    // Check if the operation is a HW module
    if (auto hwModule = dyn_cast<hw::HWModuleOp>(op)) {
      return enumerateCuts(hwModule);
    }

    // Generate cuts for this operation
    return generateCutsForOp(op);
  }

  /// Enumerate cuts for all nodes in the module
  LogicalResult enumerateCuts(hw::HWModuleOp hwModule) {
    LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

    // Topological traversal
    llvm::SmallVector<Operation *> worklist;
    llvm::DenseSet<Operation *> visited;

    // Start from inputs and constants
    hwModule.walk([&](Operation *op) {
      if (isa<hw::ConstantOp>(op) || op->getNumOperands() == 0) {
        worklist.push_back(op);
        visited.insert(op);
      }
    });

    // Process in topological order
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();

      // Generate cuts for this operation
      if (failed(generateCutsForOp(op)))
        return failure();

      // Add users to worklist if all operands processed
      for (Operation *user : op->getUsers()) {
        if (visited.count(user))
          continue;

        bool allOperandsReady = true;
        for (Value operand : user->getOperands()) {
          if (auto *defOp = operand.getDefiningOp()) {
            if (!visited.count(defOp)) {
              allOperandsReady = false;
              break;
            }
          }
        }

        if (allOperandsReady) {
          worklist.push_back(user);
          visited.insert(user);
        }
      }
    }

    return success();
  }

  /// Generate cuts for a specific operation
  LogicalResult generateCutsForOp(Operation *op) {
    Value result = op->getResult(0);
    CutSet &cutSet = cutSets[result];

    if (isa<hw::ConstantOp>(op) || op->getNumOperands() == 0) {
      // Trivial cut for inputs/constants
      Cut trivialCut(op);
      trivialCut.inputs.insert(result);
      trivialCut.area = 0.0;
      trivialCut.delay = 0.0;
      cutSet.addCut(std::move(trivialCut));
      return success();
    }

    if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
      return generateCutsForAndOp(andOp);
    }

    // For other operations, create a trivial cut
    Cut trivialCut(op);
    trivialCut.inputs.insert(result);
    trivialCut.area = 1.0;
    trivialCut.delay = 1.0;
    cutSet.addCut(std::move(trivialCut));

    return success();
  }

  /// Generate cuts for AND-inverter operations
  LogicalResult generateCutsForAndOp(aig::AndInverterOp andOp) {
    Value result = andOp.getResult();
    auto &lhs = cutSets.at(andOp->getOperand(0));
    auto &rhs = cutSets.at(andOp->getOperand(1));

    // Add trivial cut
    // Combine cuts from both inputs
    auto &resultCutSet = cutSets[result];
    for (const Cut &cut0 : lhs.getCuts()) {
      for (const Cut &cut1 : rhs.getCuts()) {
        auto mergedCut = cut0.mergeWith(cut1, andOp);
        if (mergedCut.getCutSize() > maxCutSize)
          continue; // Skip cuts that are too large
        if (mergedCut.getInputSize() > maxCutInputSize)
          continue; // Skip if merged cut exceeds max input size
        resultCutSet.addCut(std::move(mergedCut));
      }
    }

    return success();
  }

  /// Perform the actual technology mapping
  LogicalResult performMapping(hw::HWModuleOp hwModule) {
    LLVM_DEBUG(llvm::dbgs() << "Performing technology mapping...\n");

    // For now, just report the cuts found
    unsigned totalCuts = 0;
    hwModule.walk([&](aig::AndInverterOp andOp) {
      Value result = andOp.getResult();
      auto it = cutSets.find(result);
      if (it != cutSets.end()) {
        const CutSet &cutSet = it->second;
        totalCuts += cutSet.getCuts().size();

        LLVM_DEBUG({
          const Cut *bestCut = cutSet.getBestCut(compare);
          if (bestCut) {
            llvm::dbgs() << "  Node " << result << ": "
                         << cutSet.getCuts().size() << " cuts, best: "
                         << "outputSize=" << bestCut->getOutputSize()
                         << "inputSize=" << bestCut->getInputSize()
                         << " area=" << bestCut->area
                         << " delay=" << bestCut->delay << "\n";
          }
        });
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Total cuts enumerated: " << totalCuts << "\n");

    // TODO: Implement actual technology mapping transformation
    // This would involve:
    // 1. Select best cuts for each node
    // 2. Replace AIG nodes with library primitives
    // 3. Connect the mapped circuit

    return success();
  }
};

