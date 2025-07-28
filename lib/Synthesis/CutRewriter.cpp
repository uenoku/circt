//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a DAG-based boolean matching cut rewriting algorithm for
// applications like technology/LUT mapping and combinational logic
// optimization. The algorithm uses priority cuts and NPN
// (Negation-Permutation-Negation) canonical forms to efficiently match cuts
// against rewriting patterns.
//
//===----------------------------------------------------------------------===//

#include "circt/Synthesis/CutRewriter.h"

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/NPNClass.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
#include <memory>
#include <optional>

#define DEBUG_TYPE "synthesis-cut-rewriter"

using namespace circt;
using namespace circt::synthesis;

// Return true if the new area/delay is better than the old area/delay in the
// context of the given strategy.
static bool compareDelayAndArea(OptimizationStrategy strategy, double newArea,
                                ArrayRef<DelayType> newDelay, double oldArea,
                                ArrayRef<DelayType> oldDelay) {
  if (OptimizationStrategyArea == strategy) {
    // Compare by area first.
    return newArea < oldArea || (newArea == oldArea && newDelay < oldDelay);
  }
  if (OptimizationStrategyTiming == strategy) {
    // Compare by delay first.
    return newDelay < oldDelay || (newDelay == oldDelay && newArea < oldArea);
  }
  llvm_unreachable("Unknown mapping strategy");
}

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

LogicalResult
circt::synthesis::topologicallySortLogicNetwork(mlir::Operation *topOp) {

  // Sort the operations topologically
  if (topOp
          ->walk([&](Region *region) {
            auto regionKindOp =
                dyn_cast<mlir::RegionKindInterface>(region->getParentOp());
            if (!regionKindOp ||
                regionKindOp.hasSSADominance(region->getRegionNumber()))
              return WalkResult::advance();

            // Graph region.
            for (auto &block : *region) {
              if (!mlir::sortTopologically(
                      &block, [&](Value value, Operation *op) -> bool {
                        // Topologically sort AND-inverters and purely dataflow
                        // ops. Other operations can be scheduled.
                        return !(isa<aig::AndInverterOp, comb::ExtractOp,
                                     comb::ReplicateOp, comb::ConcatOp>(op));
                      }))
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return mlir::emitError(topOp->getLoc(),
                           "failed to sort operations topologically");
  return success();
}

//===----------------------------------------------------------------------===//
// Cut
//===----------------------------------------------------------------------===//

bool Cut::isPrimaryInput() const {
  // A cut is a primary input if it has no operations and only one input
  return operations.empty() && inputs.size() == 1;
}

mlir::Operation *Cut::getRoot() const {
  return operations.empty()
             ? nullptr
             : operations.back(); // The last operation is the root
}

const mlir::FailureOr<NPNClass> &Cut::getNPNClass() const {
  // If the NPN is already computed, return it
  if (npnClass)
    return *npnClass;

  auto truthTable = getTruthTable();
  if (failed(truthTable)) {
    npnClass = failure();
    return *npnClass;
  }

  // Compute the NPN canonical form
  auto canonicalForm = NPNClass::computeNPNCanonicalForm(*truthTable);

  npnClass.emplace(std::move(canonicalForm));
  return *npnClass;
}

void Cut::dump(llvm::raw_ostream &os) const {
  os << "// === Cut Dump ===\n";

  os << "Cut with " << getInputSize() << " inputs and " << getCutSize()
     << " operations:\n";
  if (isPrimaryInput()) {
    os << "Primary input cut: " << *inputs.begin() << "\n";
    return;
  }

  os << "Inputs: \n";
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    os << "  Input " << idx << ": " << input << "\n";
  }
  os << "\nOperations: \n";
  for (auto *op : operations) {
    op->print(os);
    os << "\n";
  }
  auto &npnClass = getNPNClass();
  npnClass->dump(os);

  os << "// === Cut End ===\n";
}

unsigned Cut::getInputSize() const { return inputs.size(); }

unsigned Cut::getCutSize() const { return operations.size(); }

unsigned Cut::getOutputSize() const { return getRoot()->getNumResults(); }

const llvm::FailureOr<BinaryTruthTable> &Cut::getTruthTable() const {
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
      mlir::emitError(op->getLoc(), "Failed to simulate operation") << *op;
      truthTable = failure();
      return *truthTable;
    }
    // Set the output value for the truth table
    for (unsigned j = 0; j < op->getNumResults(); ++j) {
      auto outputValue = op->getResult(j);
      if (!outputValue.getType().isInteger(1)) {
        mlir::emitError(op->getLoc(), "Output value is not a single bit: ")
            << *op;
        truthTable = failure();
        return *truthTable;
      }
    }
  }

  // Extract the truth table from the root operation
  auto rootResults = getRoot()->getResults();
  BinaryTruthTable result(numInputs, numOutputs);
  if (numOutputs == 1) {
    result.table = eval.at(rootResults[0]);
    truthTable = std::move(result);
    return *truthTable;
  }

  // If there are multiple outputs, we need to construct the truth table.
  for (unsigned input = 0; input < tableSize; ++input) {
    // For each input combination, evaluate the root operation
    APInt outputBits(numOutputs, 0);
    for (unsigned i = 0; i < numOutputs; ++i)
      outputBits.setBitVal(i, eval[rootResults[i]][input]);
    result.setOutput(APInt(numInputs, input), outputBits);
  }

  truthTable = std::move(result);
  return *truthTable;
}

static Cut getAsPrimaryInput(mlir::Value value) {
  // Create a cut with a single root operation
  Cut cut;
  // There is no input for the primary input cut.
  cut.inputs.insert(value);

  return cut;
}

static Cut getSingletonCut(mlir::Operation *op) {
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

Cut Cut::mergeWith(const Cut &other, Operation *root) const {
  // Create a new cut that combines this cut and the other cut
  Cut newCut;
  // Topological sort the operations in the new cut.
  // TODO: Merge-sort `operations` and `other.operations` by operation index
  // (since it's already topo-sorted, we can use a simple merge).
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

      // If the value is in *both* cuts inputs, it is an input. So skip
      // it.
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

LogicalResult Cut::simulateOp(Operation *op,
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
    return llvm::success();
  }
  // Add more operation types as needed
  return failure();
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

unsigned CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(std::move(cut));
}

ArrayRef<Cut> CutSet::getCuts() const { return cuts; }

void CutSet::finalize(const CutRewriterOptions &options) {
  DenseSet<std::pair<ArrayRef<Value>, Operation *>> uniqueCuts;
  unsigned uniqueCount = 0;
  for (unsigned i = 0; i < cuts.size(); ++i) {
    auto &cut = cuts[i];
    // Create a unique identifier for the cut based on its inputs.
    auto inputs = cut.inputs.getArrayRef();

    // If the cut is a duplicate, skip it.
    if (uniqueCuts.contains({inputs, cut.getRoot()}))
      continue;

    if (i != uniqueCount) {
      // Move the unique cut to the front of the vector
      // This maintains the order of cuts while removing duplicates
      // by swapping with the last unique cut found.
      cuts[uniqueCount] = std::move(cuts[i]);
    }

    // Beaware of lifetime of ArrayRef. `cuts[uniqueCount]` is always valid
    // after this point.
    uniqueCuts.insert(
        {cuts[uniqueCount].inputs.getArrayRef(), cuts[uniqueCount].getRoot()});
    ++uniqueCount;
  }

  LLVM_DEBUG(llvm::dbgs() << "Original cuts: " << cuts.size()
                          << " Unique cuts: " << uniqueCount << "\n");
  // Resize the cuts vector to the number of unique cuts found
  cuts.resize(uniqueCount);

  // Maintain size limit by removing worst cuts
  if (cuts.size() > options.maxCutSizePerRoot) {
    // Sort by priority using heuristic.
    // TODO: Make this configurable.
    std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
      return a.getCutSize() < b.getCutSize();
    });

    // TODO: Implement pruning based on dominance.

    // TODO: Pririty cuts may prune all matching cuts, so we may need to
    //       keep the matching cut before pruning.
    cuts.resize(options.maxCutSizePerRoot);
  }

  isFrozen = true; // Mark the cut set as frozen
}

//===----------------------------------------------------------------------===//
// CutEnumerator
//===----------------------------------------------------------------------===//

CutEnumerator::CutEnumerator(const CutRewriterOptions &options)
    : options(options) {}

CutSet *CutEnumerator::lookup(Value value) const {
  const auto *it = cutSets.find(value);
  if (it != cutSets.end())
    return it->second.get();
  return nullptr;
}

CutSet *CutEnumerator::createNewCutSet(Value value) {
  assert(!cutSets.contains(value) && "Cut set already exists for this value");
  auto cutSet = std::make_unique<CutSet>();
  auto *cutSetPtr = cutSet.get();
  cutSets[value] = std::move(cutSet);
  return cutSetPtr;
}

llvm::MapVector<Value, std::unique_ptr<CutSet>> CutEnumerator::takeVector() {
  return std::move(cutSets);
}

void CutEnumerator::clear() { cutSets.clear(); }

LogicalResult CutEnumerator::visit(Operation *op) {
  // For now, delegate to visitLogicOp for combinational operations
  if (isa<aig::AndInverterOp>(op))
    return visitLogicOp(op);

  // Skip non-combinational operations
  return success();
}

LogicalResult CutEnumerator::visitLogicOp(Operation *logicOp) {
  assert(logicOp->getNumResults() == 1 &&
         "Logic operation must have a single result");

  Value result = logicOp->getResult(0);
  unsigned numOperands = logicOp->getNumOperands();

  // Validate operation constraints
  // TODO: Variadic operations and non-single-bit results can be supported
  if (numOperands > 2)
    return logicOp->emitError("Cut enumeration supports at most 2 operands, "
                              "found: ")
           << numOperands;
  if (!logicOp->getOpResult(0).getType().isInteger(1))
    return logicOp->emitError("Result type must be a single bit integer");

  // Create the singleton cut (just this operation)
  Cut singletonCut = getSingletonCut(logicOp);
  auto *resultCutSet = createNewCutSet(result);

  // Add the singleton cut first
  resultCutSet->addCut(singletonCut);

  // Schedule cut set finalization when exiting this scope
  auto prune = llvm::make_scope_exit([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->finalize(options);
  });

  // Handle unary operations
  if (numOperands == 1) {
    const auto &inputCutSet = getCutSet(logicOp->getOperand(0));

    // Try to extend each input cut by including this operation
    for (const Cut &inputCut : inputCutSet.getCuts()) {
      Cut extendedCut = inputCut.mergeWith(singletonCut, logicOp);

      // Skip cuts that exceed input size limit
      if (extendedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(extendedCut));
    }
    return success();
  }

  // Handle binary operations (like AND, OR, XOR gates)
  assert(numOperands == 2 && "Expected binary operation");

  const auto &lhsCutSet = getCutSet(logicOp->getOperand(0));
  const auto &rhsCutSet = getCutSet(logicOp->getOperand(1));

  // Combine cuts from both inputs to create larger cuts
  for (const Cut &lhsCut : lhsCutSet.getCuts()) {
    for (const Cut &rhsCut : rhsCutSet.getCuts()) {
      Cut mergedCut = lhsCut.mergeWith(rhsCut, logicOp);
      // Skip cuts that exceed input size limit
      if (mergedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(mergedCut));
    }
  }

  return success();
}

LogicalResult CutEnumerator::enumerateCuts(
    Operation *topOp,
    llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts for module: " << topOp->getName()
                          << "\n");

  // Store the pattern matching function for use during cut finalization
  this->matchCut = matchCut;

  // Walk through all operations in the module in a topological manner
  auto result = topOp->walk([&](Operation *op) {
    if (failed(visit(op)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Cut enumeration completed successfully\n");
  return success();
}

const CutSet &CutEnumerator::getCutSet(Value value) {
  // Check if cut set already exists
  if (!cutSets.contains(value)) {
    // Create new cut set for primary input or unprocessed value
    cutSets[value] = std::make_unique<CutSet>();

    // Primary inputs get a trivial cut containing just themselves
    cutSets[value]->addCut(getAsPrimaryInput(value));

    LLVM_DEBUG(llvm::dbgs()
               << "Created primary input cut for: " << value << "\n");
  }

  return *cutSets.find(value)->second;
}