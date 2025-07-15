#include "circt/Synthesis/CutRewriter.h"

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

using namespace circt;
using namespace circt::synthesis;

static bool compareDelayAndArea(CutRewriteStrategy strategy, double newArea,
                                double newDelay, double oldArea,
                                double rhsDelay) {
  if (CutRewriteStrategy::Area == strategy) {
    // Compare by area only
    return newArea < oldArea || (newArea == oldArea && newDelay < rhsDelay);
  }
  if (CutRewriteStrategy::Timing == strategy) {
    // Compare by delay only
    return newDelay < rhsDelay || (newDelay == rhsDelay && newArea < oldArea);
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

void circt::synthesis::CutSet::freezeCutSet(
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
  double bestArrivalTime = std::numeric_limits<double>::max();
  double bestArea = std::numeric_limits<double>::max();

  for (auto &cut : cuts) {
    auto currentMatchedPattern =
        matchCut(cut); // Match the cut against the pattern set
    if (!currentMatchedPattern)
      continue;

    if (compareDelayAndArea(options.strategy, currentMatchedPattern->getArea(),
                            currentMatchedPattern->getArrivalTime(), bestArea,
                            bestArrivalTime)) {
      // Found a better matching pattern
      matchedPattern = currentMatchedPattern;
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
  llvm::SmallVector<mlir::Operation *, 4> worklist{root};

  // Topological sort the operations in the new cut.
  std::function<void(mlir::Operation *)> populateOperations =
      [&](mlir::Operation *op) {
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

LogicalResult CutRewriter::enumerateCuts(Operation *hwModule) {
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

/// Generate cuts for AND-inverter operations
LogicalResult CutRewriter::generateCutsForAndOp(Operation *logicOp) {
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

  Cut singletonCut = getSingletonCut(logicOp);

  auto *resultCutSet = getOrCreateCutSet(result);

  auto prune = llvm::make_scope_exit([&]() {
    // Prune the cut set to maintain the maximum number of cuts
    resultCutSet->freezeCutSet(
        options, [&](Cut &cut) { return matchCutToPattern(cut); });
  });

  // Operation itself is a cut, so add it to the cut set
  resultCutSet->addCut(singletonCut);

  if (numOperands == 1) {
    auto inputCutSet = getCutSet(logicOp->getOperand(0));
    for (const Cut &cut : inputCutSet.getCuts()) {
      // Merge the singleton cut with the input cut
      Cut newCut = cut.mergeWith(singletonCut, logicOp);
      if (newCut.getInputSize() > options.maxCutInputSize)
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
      if (mergedCut.getCutSize() > options.maxCutSizePerRoot)
        continue; // Skip cuts that are too large
      if (mergedCut.getInputSize() > options.maxCutInputSize)
        continue; // Skip if merged cut exceeds max input size
      resultCutSet->addCut(std::move(mergedCut));
    }
  }

  return success();
}
LogicalResult CutRewriter::performMapping(Operation *hwModule) {
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
  }

  return success();
}
const CutSet &CutRewriter::getCutSet(Value value) {
  // If the cut set does not exist, create a new one
  if (!cutSets.contains(value)) {
    // Add a trivial cut for primary inputs
    cutSets[value] = std::make_unique<CutSet>();
    cutSets[value]->addCut(getAsPrimaryInput(value));
  }

  return *cutSets.find(value)->second;
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
    // If the input is a primary input, it has no delay
    auto *arrivalTime = cutSets.find(input);
    if (isAlwaysCutInput(input)) {
      // If the input is a primary input, it has no delay
      inputArrivalTimes.push_back(0.0);
    } else if (arrivalTime != cutSets.end()) {
      // If the arrival time is already computed, use it
      auto pattern = arrivalTime->second->getMatchedPattern();
      if (!pattern)
        return {};
      inputArrivalTimes.push_back(pattern->getArrivalTime());
    } else {
      assert(false && "Input must have a valid arrival time");
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
        compareDelayAndArea(options.strategy, pattern->getArea(cut),
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

  if (!bestPattern)
    return std::nullopt; // No matching pattern found

  return MatchedPattern(bestPattern, &cut, bestArrivalTime);
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

CutSet *CutRewriter::getOrCreateCutSet(Value value) {
  // If the cut set does not exist, create a new one
  if (!cutSets.contains(value)) {
    // Add a trivial cut for primary inputs
    cutSets[value] = std::make_unique<CutSet>();
  }

  return cutSets.find(value)->second.get();
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

  // Process each HW module
  // Step 1: Enumerate cuts for all nodes
  if (failed(enumerateCuts(topOp)))
    return failure();

  // Step 2: Select best cuts and perform mapping
  if (failed(performMapping(topOp)))
    return failure();

  return success();
}

double MatchedPattern::getArea() const {
  assert(pattern && cut && "Pattern and cut must be set to get area");
  return pattern->getArea(*cut);
}

double MatchedPattern::getDelay(unsigned inputIndex,
                                unsigned outputIndex) const {
  assert(pattern && cut && "Pattern and cut must be set to get delay");
  return pattern->getDelay(*cut, inputIndex, outputIndex);
}