#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Synthesis/CutRewriter.h"
#include "circt/Synthesis/Transforms/Passes.h"

namespace circt {
namespace synthesis {
#define GEN_PASS_DEF_TECHMAPPER
#include "circt/Synthesis/Transforms/Passes.h.inc"
} // namespace synthesis
} // namespace circt

using namespace circt;
using namespace circt::synthesis;

//===----------------------------------------------------------------------===//
// Tech Mapper Pass
//===----------------------------------------------------------------------===//

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewriterPattern {
  hw::HWModuleOp module;

  TechLibraryPattern(hw::HWModuleOp mod) : module(mod) {}

  /// Match the cut set against this library primitive
  bool match(const Cut &cutSet) const override { return false; }

  /// Rewrite the cut set using this library primitive
  LogicalResult rewrite(mlir::PatternRewriter &rewriter,
                        Cut &cut) const override {
    cut.getCutSize();
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

      // Update the maximum input size
      maxInputSize = std::max(maxInputSize, pattern->getNumInputs());

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    CutRewriterPatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy = CutRewriteStrategy::Area; // Use area optimization
    options.maxCutInputSize = maxInputSize;
    options.maxCutSizePerRoot = maxCutsPerNode;
    CutRewriter mapper(module, options, patternSet);
    if (failed(mapper.run()))
      signalPassFailure();
  }
};

} // namespace
