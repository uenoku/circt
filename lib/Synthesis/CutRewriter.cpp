#include "circt/Synthesis/CutRewriter.h"

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
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
#include <limits>
#include <memory>
#include <optional>

#define DEBUG_TYPE "synthesis-cut-rewriter"

using namespace circt;
using namespace circt::synthesis;

// Return true if the new area/delay is better than the old area/delay in the
// context of the given strategy.
static bool compareDelayAndArea(CutRewriteStrategy strategy, double newArea,
                                double newDelay, double oldArea,
                                double oldDelay) {
  if (CutRewriteStrategy::Area == strategy) {
    // Compare by area first.
    return newArea < oldArea || (newArea == oldArea && newDelay < oldDelay);
  }
  if (CutRewriteStrategy::Timing == strategy) {
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

void CutSet::freezeCutSet(
    const CutRewriterOptions &options,
    llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut) {
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

      // Beaware of lifetime of ArrayRef. `cuts[uniqueCount]` is always valid
      // after this point.
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
  if (cuts.size() > options.maxCutSizePerRoot) {
    // Sort by priority using heuristic.
    // TODO: Make this configurable.
    std::sort(cuts.begin(), cuts.end(), [](const Cut &a, const Cut &b) {
      return a.getCutSize() < b.getCutSize();
    });
    cuts.resize(options.maxCutSizePerRoot);
    // TODO: Pririty cuts may prune all matching cuts, so we may need to
    //       keep the matching cut before pruning.
  }

  // Find the best matching pattern for this cut set
  for (auto &cut : cuts) {
    // Match the cut against the pattern set
    auto matchResult = matchCut(cut);
    if (!matchResult)
      continue;

    if (!matchedPattern ||
        compareDelayAndArea(options.strategy, matchResult->getArea(),
                            matchResult->getArrivalTime(),
                            matchedPattern->getArea(),
                            matchedPattern->getArrivalTime())) {
      // Found a better matching pattern
      matchedPattern = matchResult;
    }
  }

  isFrozen = true; // Mark the cut set as frozen
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
  llvm::SmallVector<Operation *, 4> worklist{root};

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

LogicalResult
circt::synthesis::Cut::simulateOp(Operation *op,
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
  return llvm::failure();
}

//===----------------------------------------------------------------------===//
// TruthTable Implementation
//===----------------------------------------------------------------------===//

llvm::APInt TruthTable::getOutput(const llvm::APInt &input) const {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  return table.extractBits(numOutputs, input.getZExtValue() * numOutputs);
}

void TruthTable::setOutput(const llvm::APInt &input,
                           const llvm::APInt &output) {
  assert(input.getBitWidth() == numInputs && "Input width mismatch");
  assert(output.getBitWidth() == numOutputs && "Output width mismatch");
  unsigned offset = input.getZExtValue() * numOutputs;
  for (unsigned i = 0; i < numOutputs; ++i)
    table.setBitVal(offset + i, output[i]);
}

TruthTable TruthTable::applyPermutation(
    const llvm::SmallVectorImpl<unsigned> &permutation) const {
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

TruthTable TruthTable::applyInputNegation(unsigned mask) const {
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

TruthTable TruthTable::applyOutputNegation(unsigned negation) const {
  TruthTable result(numInputs, numOutputs);

  for (unsigned i = 0; i < (1u << numInputs); ++i) {
    llvm::APInt input(numInputs, i);
    llvm::APInt output = getOutput(input);
    llvm::APInt negatedOutput(numOutputs, 0);

    // Apply negation
    for (unsigned j = 0; j < numOutputs; ++j)
      negatedOutput.setBitVal(j,
                              (negation & (1u << j)) ? !output[j] : output[j]);

    result.setOutput(input, negatedOutput);
  }

  return result;
}

bool TruthTable::isLexicographicallySmaller(const TruthTable &other) const {
  assert(numInputs == other.numInputs && numOutputs == other.numOutputs);
  return table.ult(other.table);
}

bool TruthTable::operator==(const TruthTable &other) const {
  return numInputs == other.numInputs && numOutputs == other.numOutputs &&
         table == other.table;
}

//===----------------------------------------------------------------------===//
// NPNClass Implementation
//===----------------------------------------------------------------------===//

NPNClass NPNClass::computeNPNCanonicalForm(const TruthTable &tt) {
  NPNClass canonical(tt);

  // Initialize permutation and negation vectors
  canonical.inputPermutation.resize(tt.numInputs);

  // Initialize identity permutation
  for (unsigned i = 0; i < tt.numInputs; ++i)
    canonical.inputPermutation[i] = i;

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
      unsigned currentNegMask = 0;
      for (unsigned i = 0; i < tt.numInputs; ++i) {
        // Permute the negation mask according to the permutation
        if (negMask & (1u << i)) {
          currentNegMask |= (1u << permutation[i]);
        } else {
          currentNegMask &= ~(1u << permutation[i]);
        }
      }

      // Try output negation (for single output)
      if (tt.numOutputs == 1) {
        unsigned outputNegMask = 0;
        TruthTable candidate = permutedTT;

        NPNClass canonicalCandidate(permutedTT, permutation, currentNegMask, 0);

        // Try without output negation
        if (canonicalCandidate.isLexicographicallySmaller(canonical)) {
          canonical = canonicalCandidate;
        }

        // Try with output negation
        candidate = permutedTT.applyOutputNegation(1);
        NPNClass newCanonical(permutedTT.applyOutputNegation(1), permutation,
                              currentNegMask, 1);
        if (newCanonical.isLexicographicallySmaller(canonical)) {
          canonical = newCanonical;
        }
      } else {
        assert(false);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }

  return canonical;
}

//===----------------------------------------------------------------------===//
// Cut Implementation
//===----------------------------------------------------------------------===//

bool Cut::isPrimaryInput() const {
  // A cut is a primary input if it has no operations and only one input
  return operations.empty() && inputs.size() == 1;
}

mlir::Operation *Cut::getRoot() const {
  return operations.empty()
             ? nullptr
             : operations.back(); // The first operation is the root
}

const mlir::FailureOr<NPNClass> &Cut::getNPNClass() const {
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

void Cut::dump() const {
  llvm::dbgs() << "// === Cut Dump ===\n";

  llvm::dbgs() << "Cut with " << getInputSize() << " inputs and "
               << getCutSize() << " operations:\n";
  if (isPrimaryInput()) {
    llvm::dbgs() << "Primary input cut: " << *inputs.begin() << "\n";
    return;
  }

  llvm::dbgs() << "Inputs: \n";
  for (auto [idx, input] : llvm::enumerate(inputs)) {
    llvm::dbgs() << "  Input " << idx << ": " << input << "\n";
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
  llvm::dbgs() << npnClass->inputNegation << " (input negation) "
               << npnClass->outputNegation << " (output negation)\n";

  llvm::dbgs() << "// === Cut End ===\n";
}

unsigned Cut::getInputSize() const { return inputs.size(); }

unsigned Cut::getCutSize() const { return operations.size(); }

size_t Cut::getOutputSize() const { return getRoot()->getNumResults(); }

const llvm::FailureOr<TruthTable> &Cut::getTruthTable() const {
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
    for (size_t j = 0; j < op->getNumResults(); ++j) {
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
  assert(rootResults.size() == 1 &&
         "For now we only support single output cuts");
  auto result = rootResults[0];

  // Cache the truth table
  truthTable = TruthTable(numInputs, numOutputs, eval[result]);
  return *truthTable;
}

//===----------------------------------------------------------------------===//
// MatchedPattern
//===----------------------------------------------------------------------===//

double MatchedPattern::getArrivalTime() const {
  assert(pattern && "Pattern must be set to get arrival time");
  return arrivalTime;
}

CutRewriterPattern *MatchedPattern::getPattern() const {
  assert(pattern && "Pattern must be set to get the pattern");
  return pattern;
}

Cut *MatchedPattern::getCut() const {
  assert(cut && "Cut must be set to get the cut");
  return cut;
}

bool MatchedPattern::isValid() const { return pattern != nullptr; }

double MatchedPattern::getArea() const {
  assert(pattern && "Pattern must be set to get area");
  return pattern->getArea(*cut);
}

double MatchedPattern::getDelay(unsigned inputIndex,
                                unsigned outputIndex) const {
  assert(pattern && "Pattern must be set to get delay");
  return pattern->getDelay(*cut, inputIndex, outputIndex);
}

//===----------------------------------------------------------------------===//
// CutSet
//===----------------------------------------------------------------------===//

bool CutSet::isMatched() const {
  return matchedPattern.has_value() && matchedPattern->isValid();
}

double CutSet::getArrivalTime() const {
  assert(isMatched() &&
         "Matched pattern must be set before getting arrival time");
  return matchedPattern->getArrivalTime();
}

std::optional<MatchedPattern> CutSet::getMatchedPattern() const {
  return matchedPattern;
}

Cut *CutSet::getMatchedCut() {
  assert(isMatched() &&
         "Matched pattern must be set before getting matched cut");
  return matchedPattern->getCut();
}

size_t CutSet::size() const { return cuts.size(); }

void CutSet::addCut(Cut cut) {
  assert(!isFrozen && "Cannot add cuts to a frozen cut set");
  cuts.push_back(std::move(cut));
}

ArrayRef<Cut> CutSet::getCuts() const { return cuts; }

//===----------------------------------------------------------------------===//
// CutRewriterPattern
//===----------------------------------------------------------------------===//

bool CutRewriterPattern::useTruthTableMatcher(
    SmallVectorImpl<NPNClass> &matchingNPNClasses) const {
  return false;
}

//===----------------------------------------------------------------------===//
// CutRewriterPatternSet
//===----------------------------------------------------------------------===//

CutRewriterPatternSet::CutRewriterPatternSet(
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

//===----------------------------------------------------------------------===//
// CutRewriter
//===----------------------------------------------------------------------===//

LogicalResult CutRewriter::sortOperationsTopologically(Operation *hwModule) {
  // Sort the operations topologically
  if (hwModule
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
    return mlir::emitError(hwModule->getLoc(),
                           "failed to sort operations topologically");
  return success();
}

LogicalResult CutRewriter::run() {
  LLVM_DEBUG({
    llvm::dbgs() << "Starting Cut Rewriter\n";
    llvm::dbgs() << "Mode: "
                 << (CutRewriteStrategy::Area == options.strategy ? "area"
                                                                  : "timing")
                 << "\n";
    llvm::dbgs() << "Max cut size: " << options.maxCutSizePerRoot << "\n";
    llvm::dbgs() << "Max cuts per node: " << options.maxCutSizePerRoot << "\n";
  });

  // First sort the operations topologically to ensure we can process them
  // in a valid order.
  if (failed(sortOperationsTopologically(topOp)))
    return failure();

  // Enumerate cuts for all nodes
  if (failed(enumerateCuts(topOp)))
    return failure();

  // Select best cuts and perform mapping
  if (failed(performRewriting(topOp)))
    return failure();

  return success();
}

LogicalResult CutRewriter::enumerateCuts(Operation *hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts...\n");

  return cutEnumerator.enumerateCuts(
      hwModule, [&](Cut &cut) -> std::optional<MatchedPattern> {
        // Match the cut against the patterns
        return matchCutToPattern(cut);
      });
}

LogicalResult CutEnumerator::visitLogicOp(Operation *logicOp) {
  assert(logicOp->getNumResults() == 1 &&
         "Logic operation must have a single result");

  Value result = logicOp->getResult(0);
  size_t numOperands = logicOp->getNumOperands();

  // Validate operation constraints
  if (numOperands > 2) {
    return logicOp->emitError("Cut enumeration supports at most 2 operands, "
                              "found: ")
           << numOperands;
  }

  if (!logicOp->getOpResult(0).getType().isInteger(1)) {
    return logicOp->emitError("Result type must be a single bit integer");
  }

  // Create the singleton cut (just this operation)
  Cut singletonCut = getSingletonCut(logicOp);
  auto *resultCutSet = createNewCutSet(result);

  // Add the singleton cut first
  resultCutSet->addCut(singletonCut);

  // Schedule cut set finalization when exiting this scope
  auto prune = llvm::make_scope_exit([&]() {
    // Finalize cut set: remove duplicates, limit size, and match patterns
    resultCutSet->freezeCutSet(options, matchCut);
  });

  // Handle unary operations (like NOT gates)
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

      // Skip cuts that exceed size limits
      if (mergedCut.getCutSize() > options.maxCutSizePerRoot)
        continue;
      if (mergedCut.getInputSize() > options.maxCutInputSize)
        continue;

      resultCutSet->addCut(std::move(mergedCut));
    }
  }

  return success();
}

LogicalResult CutEnumerator::enumerateCuts(
    Operation *hwModule,
    llvm::function_ref<std::optional<MatchedPattern>(Cut &)> matchCut) {
  LLVM_DEBUG(llvm::dbgs() << "Enumerating cuts for module: "
                          << hwModule->getName() << "\n");

  // Store the pattern matching function for use during cut finalization
  this->matchCut = matchCut;

  // Walk through all operations in the module in a topological manner
  auto result = hwModule->walk([&](Operation *op) {
    if (failed(visit(op)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted())
    return mlir::emitError(hwModule->getLoc(),
                           "Failed to enumerate cuts for module");

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

ArrayRef<std::pair<NPNClass, CutRewriterPattern *>>
CutRewriter::getMatchingPatternFromTruthTable(const Cut &cut) const {
  if (patterns.npnToPatternMap.empty())
    return {};

  auto &npnClass = cut.getNPNClass();
  auto it = patterns.npnToPatternMap.find(npnClass->truthTable.table);
  if (it == patterns.npnToPatternMap.end())
    return {};
  return it->getSecond();
}

std::optional<MatchedPattern> CutRewriter::matchCutToPattern(Cut &cut) {
  if (cut.isPrimaryInput())
    return {};

  double bestArrivalTime = std::numeric_limits<double>::max();
  CutRewriterPattern *bestPattern = nullptr;
  SmallVector<double, 4> inputArrivalTimes, outputArrivalTimes;
  inputArrivalTimes.reserve(cut.getInputSize());
  outputArrivalTimes.reserve(cut.getOutputSize());
  for (auto input : cut.inputs) {
    assert(input.getType().isInteger(1));
    if (isAlwaysCutInput(input)) {
      // If the input is a primary input, it has no delay
      inputArrivalTimes.push_back(0.0);
      continue;
    }
    auto *cutSet = cutEnumerator.lookup(input);
    if (cutSet) {
      // If the arrival time is already computed, use it
      auto pattern = cutSet->getMatchedPattern();
      if (!pattern)
        return {};
      inputArrivalTimes.push_back(pattern->getArrivalTime());
    } else {
      assert(false && "Input must have a valid arrival time");
    }
  }

  auto computeArrivalTimeAndPickBest =
      [&](CutRewriterPattern *pattern,
          llvm::function_ref<unsigned(unsigned)> mapInput) {
        // If the pattern matches the cut, compute the arrival time
        double patternArrivalTime = 0.0;

        // Compute the maximum delay for each output from inputs
        for (size_t i = 0; i < cut.getInputSize(); ++i) {
          for (size_t j = 0; j < cut.getOutputSize(); ++j) {
            // Map pattern input i to cut input through NPN transformations
            unsigned cutOriginalInput = mapInput(i);
            patternArrivalTime =
                std::max(patternArrivalTime,
                         pattern->getDelay(cut, cutOriginalInput, j) +
                             inputArrivalTimes[cutOriginalInput]);
          }
        }

        if (!bestPattern ||
            compareDelayAndArea(options.strategy, pattern->getArea(cut),
                                patternArrivalTime, bestPattern->getArea(cut),
                                bestArrivalTime)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Picking pattern: " << pattern->getPatternName()
                     << " with arrival time: " << patternArrivalTime
                     << " and area: " << pattern->getArea(cut) << "\n");
          bestArrivalTime = patternArrivalTime;
          bestPattern = pattern;
        }
      };

  for (auto &[patternNPN, pattern] : getMatchingPatternFromTruthTable(cut)) {
    if (!pattern->match(cut))
      continue;
    auto &cutNPN = cut.getNPNClass();

    // Build inverse permutation mapping from cut's canonical form to
    // original cut inputs
    // TODO: Cache permutation/inv-permutation via unique id.
    llvm::SmallVector<unsigned> cutInversePermutation(cut.getInputSize());
    for (size_t i = 0; i < cut.getInputSize(); ++i) {
      cutInversePermutation[cutNPN->inputPermutation[i]] = i;
    }

    computeArrivalTimeAndPickBest(
        pattern, [&](unsigned i) { return cutInversePermutation[i]; });
  }

  for (CutRewriterPattern *pattern : patterns.nonTruthTablePatterns)
    if (pattern->match(cut))
      computeArrivalTimeAndPickBest(pattern, [&](unsigned i) { return i; });

  if (!bestPattern)
    return std::nullopt; // No matching pattern found

  return MatchedPattern(bestPattern, &cut, bestArrivalTime);
}

LogicalResult CutRewriter::performRewriting(Operation *hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "Performing cut-based rewriting...\n");

  // For now, just report the cuts found
  unsigned totalCuts = 0;
  LLVM_DEBUG(llvm::dbgs() << "Total cuts enumerated: " << totalCuts << "\n");

  // TODO: Implement actual cut-based rewriting transformation
  // This would involve:
  // 1. Select best cuts for each node
  // 2. Replace AIG nodes with library primitives
  // 3. Connect the mapped circuit
  auto cutVector = cutEnumerator.takeVector();
  UnusedOpPruner pruner;
  PatternRewriter rewriter(hwModule->getContext());
  double maximumArrivalTime = 0.0;
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
    auto matchedPattern = cutSet->getMatchedPattern();
    if (!matchedPattern) {
      return mlir::emitError(hwModule->getLoc(),
                             "No matching cut found for value: ")
             << value;
    }

    auto *cut = matchedPattern->getCut();
    rewriter.setInsertionPoint(cut->getRoot());
    if (failed(matchedPattern->getPattern()->rewrite(rewriter, *cut))) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Rewrote cut for value: " << value << " with pattern: "
               << matchedPattern->getPattern()->getPatternName() << "\n");
    // Update maximum arrival time
    maximumArrivalTime =
        std::max(maximumArrivalTime, matchedPattern->getArrivalTime());
  }

  // If we have a maximum arrival time, report it
  LLVM_DEBUG({
    llvm::dbgs() << "Maximum arrival time after rewriting: "
                 << maximumArrivalTime << "\n";
  });

  return success();
}

//===----------------------------------------------------------------------===//
// CutEnumerator Implementation
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
