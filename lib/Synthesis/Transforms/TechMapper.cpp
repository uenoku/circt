#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/NPNClass.h"
#include "circt/Synthesis/CutRewriter.h"
#include "circt/Synthesis/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace circt {
namespace synthesis {
#define GEN_PASS_DEF_TECHMAPPER
#include "circt/Synthesis/Transforms/Passes.h.inc"
} // namespace synthesis
} // namespace circt

using namespace circt;
using namespace circt::synthesis;

#define DEBUG_TYPE "synthesis-tech-mapper"

//===----------------------------------------------------------------------===//
// Tech Mapper Pass
//===----------------------------------------------------------------------===//

static LogicalResult simulateOp(Operation *op, DenseMap<Value, APInt> &values) {
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op)) {
    SmallVector<APInt> inputs;
    for (auto input : andOp.getInputs()) {
      auto it = values.find(input);
      if (it == values.end())
        return op->emitError("Input value not found in evaluation map");
      inputs.push_back(it->second);
    }
    values[andOp.getResult()] = andOp.evaluate(inputs);
    return success();
  }
  // Add more operation types as needed.
  return op->emitError("Unsupported operation for truth table generation");
}

static llvm::FailureOr<NPNClass> getNPNClassFromModule(hw::HWModuleOp module) {
  // Get input and output ports
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();

  unsigned numInputs = inputTypes.size();
  unsigned numOutputs = outputTypes.size();

  // Verify all ports are single bit
  for (auto type : inputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All input ports must be single bit");
  }
  for (auto type : outputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All output ports must be single bit");
  }

  if (numInputs >= 16)
    return module->emitError("Too many inputs for truth table generation");

  // Create truth table
  uint32_t tableSize = 1 << numInputs;
  DenseMap<Value, APInt> eval;

  // Set up input values for all possible input combinations
  auto inputArgs = module.getBodyBlock()->getArguments();
  for (unsigned i = 0; i < numInputs; ++i) {
    APInt value(tableSize, 0);
    for (unsigned j = 0; j < tableSize; ++j) {
      // Set bit j to the i-th bit of the input combination j
      value.setBitVal(j, (j >> i) & 1);
    }
    eval[inputArgs[i]] = std::move(value);
  }

  // Collect all operations in topological order
  SmallVector<Operation *> operations;
  if (failed(topologicallySortLogicNetwork(module)))
    return module->emitError("Failed to sort operations topologically");

  auto result = module.walk([&](Operation *op) {
    // Skip non-combinational operations
    if (!isa<aig::AndInverterOp>(op))
      return mlir::WalkResult::advance();

    // Simulate the operation
    if (failed(simulateOp(op, eval))) {
      op->emitError("Failed to simulate operation in module");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted())
    return module->emitError("Failed to simulate all operations");

  // Extract outputs from the hw.output operation
  auto outputOp = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
  SmallVector<APInt> outputs;
  for (unsigned i = 0; i < numOutputs; ++i) {
    auto outputValue = outputOp.getOperand(i);
    auto it = eval.find(outputValue);
    if (it == eval.end())
      return mlir::emitError(outputValue.getLoc(),
                             "Failed to evaluate a truth table");
    outputs.push_back(eval.at(outputValue));
  }

  // Create TruthTable and compute NPN canonical form
  BinaryTruthTable truthTable(numInputs, numOutputs, outputs);
  return NPNClass::computeNPNCanonicalForm(truthTable);
}

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<SmallVector<DelayType, 2>, 4> delay,
                     NPNClass npnClass)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), npnClass(std::move(npnClass)) {

    LLVM_DEBUG(
        llvm::dbgs() << "Created Tech Library Pattern for module: "
                     << module.getModuleName() << "\n";
        llvm::dbgs() << "NPN Class: " << npnClass.truthTable.table << "\n";
        llvm::dbgs() << "Inputs: " << npnClass.inputPermutation.size() << "\n";
        llvm::dbgs() << "Input Negation: " << npnClass.inputNegation << "\n";
        llvm::dbgs() << "Output Negation: " << npnClass.outputNegation
                     << "\n";);
  }

  StringRef getPatternName() const override {
    auto moduleCp = module;
    return moduleCp.getModuleName();
  }

  /// Match the cut set against this library primitive
  bool match(const Cut &cut) const override {
    return cut.getNPNClass()->equivalentOtherThanPermutation(npnClass);
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &rewriter,
                                       Cut &cut) const override {
    // Create a new instance of the module
    SmallVector<Value> inputs;
    cut.getMappedInputs(npnClass, inputs);

    // TODO: Give a better name to the instance
    auto instanceOp = rewriter.create<hw::InstanceOp>(
        cut.getRoot()->getLoc(), module, "mapped", ArrayRef<Value>(inputs));
    return instanceOp.getOperation();
  }

  double getArea(const Cut &cut) const override { return area; }

  DelayType getDelay(unsigned inputIndex, unsigned outputIndex) const override {
    return delay[inputIndex][outputIndex];
  }

  unsigned getNumInputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumInputPorts();
  }

  unsigned getNumOutputs() const override {
    return static_cast<hw::HWModuleOp>(module).getNumOutputPorts();
  }

  LocationAttr getLoc() const override {
    auto module = this->module;
    return module.getLoc();
  }

private:
  const double area;
  const SmallVector<SmallVector<DelayType, 2>, 4> delay;
  hw::HWModuleOp module;
  NPNClass npnClass;
};

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;

    unsigned maxInputSize = 0;
    // Consider modules with the "hw.techlib.info" attribute as library
    // modules.
    SmallVector<hw::HWModuleOp> nonLibraryModules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      auto techInfo =
          hwModule->getAttrOfType<DictionaryAttr>("hw.techlib.info");
      if (!techInfo) {
        // If the module does not have the techlib info, it is not a library
        // TODO: Run mapping only when the module is under the specific
        // hierarchy.
        nonLibraryModules.push_back(hwModule);
        continue;
      }

      // Get area and delay attributes
      auto areaAttr = techInfo.getAs<FloatAttr>("area");
      auto delayAttr = techInfo.getAs<ArrayAttr>("delay");
      if (!areaAttr || !delayAttr) {
        mlir::emitError(hwModule.getLoc())
            << "Library module " << hwModule.getModuleName()
            << " must have 'area'(float) and 'delay' (2d array to represent "
               "input-output pair delay) attributes";
        signalPassFailure();
        return;
      }

      double area = areaAttr.getValue().convertToDouble();

      SmallVector<SmallVector<DelayType, 2>, 4> delay;
      for (auto delayValue : delayAttr) {
        auto delayArray = cast<ArrayAttr>(delayValue);
        SmallVector<DelayType, 2> delayRow;
        for (auto delayElement : delayArray) {
          // FIXME: Currently we assume delay is given as integer attributes,
          // this should be replaced once we have a proper cell op with
          // dedicated timing attributes with units.
          delayRow.push_back(
              cast<mlir::IntegerAttr>(delayElement).getValue().getZExtValue());
        }
        delay.push_back(std::move(delayRow));
      }
      // Compute NPN Class for the module.
      auto npnClass = getNPNClassFromModule(hwModule);
      if (failed(npnClass)) {
        signalPassFailure();
        return;
      }

      // Create a CutRewritePattern for the library module
      std::unique_ptr<CutRewritePattern> pattern =
          std::make_unique<TechLibraryPattern>(hwModule, area, std::move(delay),
                                               std::move(*npnClass));

      // Update the maximum input size
      maxInputSize = std::max(maxInputSize, pattern->getNumInputs());

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    if (libraryPatterns.empty())
      return markAllAnalysesPreserved();

    CutRewritePatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.attachDebugTiming = test;
    auto result = mlir::failableParallelForEach(
        module.getContext(), nonLibraryModules, [&](hw::HWModuleOp hwModule) {
          LLVM_DEBUG(llvm::dbgs() << "Processing non-library module: "
                                  << hwModule.getName() << "\n");
          CutRewriter rewriter(options, patternSet);
          return rewriter.run(hwModule);
        });
    if (failed(result))
      signalPassFailure();
  }
};

} // namespace
