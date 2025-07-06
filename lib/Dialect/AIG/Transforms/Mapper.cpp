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
  APInt table;         // Truth table represented as an APInt
  unsigned numInputs;  // Number of inputs for this cut
  unsigned numOutputs; // Number of outputs for this cut
  TruthTable(unsigned numInputs, unsigned numOutputs)
      : numInputs(numInputs), numOutputs(numOutputs),
        table((1 << numInputs) * numOutputs, 0) {}

  /// Get the output value for a given input combination
  APInt getOutput(const APInt &input) const {
    assert(input.getBitWidth() == numInputs && "Input width mismatch");
    return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
  }
};

} // end anonymous namespace

/// Represents a cut in the AIG network
struct Cut {
  llvm::SmallSetVector<Value, 4> inputs; // Cut inputs (leaves)
  // Operations in the cut. Operation is topologcically sorted, with the root
  // operation at the front.
  llvm::SmallSetVector<Operation *, 4> operations;

  double delay = 0; // Delay of the cut
  double area = 0;  // Area of the cut

  Cut() = default;

  Operation *getRoot() const {
    return operations.front(); // The last operation is the root
  }

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
  unsigned getInputSize() const {}
  unsigned getCutSize() const { return operations.size(); }
  unsigned getOutputSize() const {
    // The output size is the number of results of the root operation
    return getRoot()->getNumResults();
  }

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

  TruthTable getTruthTable() {
    TruthTable tt(inputs.size(), getOutputSize());
    // Simulate the IR.
    // For each input combination, compute the output values.
    for (uint64_t i = 0; i < (1 << inputs.size()); ++i) {
      // Compute the input vector
      APInt input(inputs.size(), i);
    }
    return tt;
  }
};

/// Cut set for a node using priority cuts algorithm
class CutSet {
private:
  llvm::SmallVector<Cut, 12> cuts;

public:
  // Keep the cuts sorted by priority. The priority is determined by the
  // optimization mode (area or delay). The cuts are sorted in descending order,
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
struct MappedLibrary {
  virtual ~MappedLibrary() = default;

  // Run heavy initialization logic here if needed.
  virtual void initialize() {};

  // Return true if the cut matches this library primitive.
  virtual bool match(const Cut &cutSet) const = 0;

  // Rewrite the cut.
  virtual LogicalResult rewrite(mlir::PatternRewriter &rewriter,
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
struct LibraryPrimitive : public MappedLibrary {
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

struct GenericLUT : public MappedLibrary {
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
    llvm::report_fatal_error("GenericLUT::rewrite not implemented yet");
    auto truthTable = cut.getTruthTable();
    // Generate comb.truth table operation.
    auto truthTableOp = rewriter.create<comb::TruthTableOp>(
        cut.getRoot()->getLoc(), truthTable.table, truthTable.numInputs,
        truthTable.numOutputs);

    // Replace the root operation with the truth table operation
    rewriter.replaceOp(cut.getRoot(), truthTableOp.getResults());
  }
};

/// Main mapper class
class AIGMapper {
private:
  ModuleOp topModule;
  bool areaMode;
  unsigned maxCutSize;
  unsigned maxCutInputSize = 6; // Default max cut input size
  unsigned maxCutsPerNode;
  bool verbose;
  bool enableTiming;

  // Technology library
  llvm::SmallVector<std::unique_ptr<MappedLibrary>> library;
  llvm::DenseMap<Value, CutSet> cutSets;
  llvm::DenseMap<Value, double> arrivalTimes;
  std::function<bool(const Cut &, const Cut &)> compare;

public:
  AIGMapper(ModuleOp module, bool areaMode, unsigned maxCutSize,
            unsigned maxCutsPerNode, bool verbose, bool enableTiming,
            ArrayRef<std::string> libraryModules)
      : topModule(module), areaMode(areaMode), maxCutSize(maxCutSize),
        maxCutsPerNode(maxCutsPerNode), verbose(verbose),
        enableTiming(enableTiming) {
    // Set the comparison function based on area or delay mode
    if (areaMode) {
      compare = [](const Cut &a, const Cut &b) {
        return a.area < b.area || (a.area == b.area && a.delay < b.delay);
      };
    } else {
      compare = [](const Cut &a, const Cut &b) {
        return a.delay < b.delay || (a.delay == b.delay && a.area < b.area);
      };
    }

    // Initialize library
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
          std::unique_ptr<MappedLibrary> it =
              std::unique_ptr<MappedLibrary>(new LibraryPrimitive(hwModule));
          library.emplace_back(std::move(it));
          if (verbose) {
            llvm::outs() << "Added library module: " << moduleName;
          }
        }
      });
    }
  }

  /// Run the mapping algorithm
  LogicalResult runMapper() {
    if (verbose) {
      llvm::outs() << "Starting AIG technology mapping\n";
      llvm::outs() << "Mode: " << (areaMode ? "area" : "delay") << "\n";
      llvm::outs() << "Max cut size: " << maxCutSize << "\n";
      llvm::outs() << "Max cuts per node: " << maxCutsPerNode << "\n";
    }

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
    if (verbose) {
      llvm::outs() << "Mapping module: " << hwModule.getModuleName() << "\n";
    }

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
    if (verbose) {
      llvm::outs() << "Enumerating cuts for operation: " << op->getName()
                   << "\n";
    }

    // Check if the operation is a HW module
    if (auto hwModule = dyn_cast<hw::HWModuleOp>(op)) {
      return enumerateCuts(hwModule);
    }

    // Generate cuts for this operation
    return generateCutsForOp(op);
  }

  /// Enumerate cuts for all nodes in the module
  LogicalResult enumerateCuts(hw::HWModuleOp hwModule) {
    if (verbose) {
      llvm::outs() << "Enumerating cuts...\n";
    }

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
          if (auto defOp = operand.getDefiningOp()) {
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
    if (verbose) {
      llvm::outs() << "Performing technology mapping...\n";
    }

    // For now, just report the cuts found
    unsigned totalCuts = 0;
    hwModule.walk([&](aig::AndInverterOp andOp) {
      Value result = andOp.getResult();
      auto it = cutSets.find(result);
      if (it != cutSets.end()) {
        const CutSet &cutSet = it->second;
        totalCuts += cutSet.getCuts().size();

        if (verbose) {
          const Cut *bestCut = cutSet.getBestCut(compare);
          if (bestCut) {
            llvm::outs() << "  Node " << result << ": "
                         << cutSet.getCuts().size() << " cuts, best: "
                         << "outputSize=" << bestCut->getOutputSize()
                         << "inputSize=" << bestCut->getInputSize()
                         << " area=" << bestCut->area
                         << " delay=" << bestCut->delay << "\n";
          }
        }
      }
    });

    if (verbose) {
      llvm::outs() << "Total cuts enumerated: " << totalCuts << "\n";
    }

    // TODO: Implement actual technology mapping transformation
    // This would involve:
    // 1. Select best cuts for each node
    // 2. Replace AIG nodes with library primitives
    // 3. Connect the mapped circuit

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

    AIGMapper mapper(module, areaMode, maxCutSize, maxCutsPerNode, verbose,
                     enableTiming, libraryModules);

    if (failed(mapper.runMapper())) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace
