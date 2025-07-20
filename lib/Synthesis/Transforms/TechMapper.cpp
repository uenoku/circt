#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Synthesis/CutRewriter.h"
#include "circt/Synthesis/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

static LogicalResult simulateHWOp(Operation *op,
                                  DenseMap<Value, APInt> &values) {
  // Simulate AndInverter op.

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
  // Add more operation types as needed
  return op->emitError("Unsupported operation for truth table generation");
}

static NPNClass getNPNClassFromModule(hw::HWModuleOp module) {
  // Get input and output ports
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();

  unsigned numInputs = inputTypes.size();
  unsigned numOutputs = outputTypes.size();

  // Verify all ports are single bit
  for (auto type : inputTypes) {
    if (!type.isInteger(1)) {
      module->emitError("All input ports must be single bit");
      return NPNClass();
    }
  }
  for (auto type : outputTypes) {
    if (!type.isInteger(1)) {
      module->emitError("All output ports must be single bit");
      return NPNClass();
    }
  }

  if (numInputs >= 20) {
    module->emitError("Too many inputs for truth table generation");
    return NPNClass();
  }

  // Create truth table
  uint32_t tableSize = 1 << numInputs;
  DenseMap<Value, APInt> eval;

  // Set up input values for all possible input combinations
  auto inputArgs = module.getBodyBlock()->getArguments();
  for (unsigned i = 0; i < numInputs; ++i) {
    APInt value(tableSize, 0);
    for (uint32_t j = 0; j < tableSize; ++j) {
      // Set bit j to the i-th bit of the input combination j
      value.setBitVal(j, (j >> i) & 1);
    }
    eval[inputArgs[i]] = std::move(value);
  }

  // Collect all operations in topological order
  SmallVector<Operation *> operations;
  for (auto &op : module.getBodyBlock()->getOperations()) {
    if (!isa<hw::OutputOp>(op)) {
      operations.push_back(&op);
    }
  }

  // Simulate all operations
  for (auto *op : operations) {
    if (failed(simulateHWOp(op, eval))) {
      module->emitError("Failed to simulate operation in module");
      return NPNClass();
    }
  }

  // Extract outputs from the hw.output operation
  auto outputOp = cast<hw::OutputOp>(module.getBodyBlock()->getTerminator());
  APInt truthTableBits(tableSize * numOutputs, 0);

  for (unsigned i = 0; i < numOutputs; ++i) {
    auto outputValue = outputOp.getOperand(i);
    auto it = eval.find(outputValue);
    if (it == eval.end()) {
      module->emitError("Output value not found in evaluation");
      return NPNClass();
    }

    // Pack output bits into truth table
    for (unsigned j = 0; j < tableSize; ++j) {
      if (it->second[j]) {
        truthTableBits.setBitVal(j * numOutputs + i, true);
      }
    }
  }

  // Create TruthTable and compute NPN canonical form
  TruthTable truthTable(numInputs, numOutputs, truthTableBits);
  return NPNClass::computeNPNCanonicalForm(truthTable);
}

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewriterPattern {
  hw::HWModuleOp module;
  NPNClass npnClass;

  TechLibraryPattern(hw::HWModuleOp mod) : module(mod) {
    // Create an NPN class from the module's truth table
    npnClass = getNPNClassFromModule(module);
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
    
    // Get both NPN classes to compose their permutations
    const auto &cutNPN = *cut.getNPNClass();
    const auto &patternNPN = npnClass;
    
    // Compute the composed permutation that maps from module input positions
    // to cut input positions. This accounts for both the cut's canonical form
    // and the pattern's canonical form.
    auto cutInversePermutation = NPNClass::invertPermutation(cutNPN.inputPermutation);
    
    // Alternative approach using composePermutations:
    // auto directMapping = NPNClass::composePermutations(cutInversePermutation, patternNPN.inputPermutation);
    // Then: inputs.push_back(cut.inputs[directMapping[i]]);
    
    // For each module input position, find the corresponding cut input
    for (unsigned i = 0; i < cut.getInputSize(); ++i) {
      // Module input i corresponds to canonical position patternNPN.inputPermutation[i]
      // We need the cut input that maps to the same canonical position
      unsigned canonicalPos = patternNPN.inputPermutation[i];
      unsigned cutInputIndex = cutInversePermutation[canonicalPos];
      inputs.push_back(cut.inputs[cutInputIndex]);
    }

    auto instanceOp = rewriter.create<hw::InstanceOp>(
        cut.getRoot()->getLoc(), module, "mapped", ArrayRef<Value>(inputs));
    return instanceOp.getOperation();
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

  DelayType getDelay(const Cut &cut, size_t inputIndex,
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

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  void runOnOperation() override {
    auto module = getOperation();

    if (libraryModules.empty())
      return markAllAnalysesPreserved();

    auto &symbolTable = getAnalysis<SymbolTable>();
    SmallVector<std::unique_ptr<CutRewriterPattern>> libraryPatterns;

    unsigned maxInputSize = 0;
    llvm::StringSet<> libraryModuleSet;

    // Find library modules in the top module
    for (const std::string &moduleName : libraryModules) {
      libraryModuleSet.insert(moduleName);

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

      // Update the maximum input size
      maxInputSize = std::max(maxInputSize, pattern->getNumInputs());

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    CutRewriterPatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy =
        strategy == synthesis::OptimizationStrategyArea
            ? CutRewriteStrategy::Area
            : CutRewriteStrategy::Timing; // Use area optimization
    options.maxCutInputSize = maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerNode;
    options.attachDebugTiming = true;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Processing HW module: " << hwModule.getName() << "\n");
      if (libraryModuleSet.contains(hwModule.getModuleName()))
        continue; // Skip if this module is a library module

      CutRewriter mapper(options, patternSet);
      if (failed(mapper.run(hwModule)))
        signalPassFailure();
    }
  }
};

} // namespace
