#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Synthesis/CutRewriter.h"
#include "circt/Synthesis/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synthesis-generic-lut-mapper"

namespace circt {
namespace synthesis {
#define GEN_PASS_DEF_GENERICLUTMAPPER
#include "circt/Synthesis/Transforms/Passes.h.inc"
} // namespace synthesis
} // namespace circt

using namespace circt;
//===----------------------------------------------------------------------===//
// Generic LUT Mapper Pass
//===----------------------------------------------------------------------===//
using namespace circt;
using namespace circt::synthesis;
namespace {

struct GenericLUT : public CutRewritePattern {
  /// Generic LUT primitive with k inputs
  unsigned k; // Number of inputs for the LUT
  GenericLUT(mlir::MLIRContext *context, unsigned k)
      : CutRewritePattern(context), k(k) {}
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

  DelayType getDelay(unsigned inputIndex, unsigned outputIndex) const override {
    // Assume a fixed delay for the generic LUT
    return 1.0;
  }

  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &rewriter,
                                       Cut &cut) const override {
    // NOTE: Don't use NPN since it's unnecessary.
    auto truthTable = cut.getTruthTable();
    if (failed(truthTable))
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting cut with " << cut.getInputSize()
                   << " inputs and " << cut.getCutSize()
                   << " operations to a generic LUT with " << k << " inputs.\n";
      cut.dump(llvm::dbgs());
      llvm::dbgs() << "Truth table details:\n";
      truthTable->dump(llvm::dbgs());
    });

    SmallVector<bool> lutTable;
    // Convert the truth table to a LUT table
    for (uint32_t i = 0; i < truthTable->table.getBitWidth(); ++i)
      lutTable.push_back(truthTable->table[i]);

    auto arrayAttr = rewriter.getBoolArrayAttr(
        lutTable); // Create a boolean array attribute.

    // Reverse the inputs to match the LUT input order
    SmallVector<Value> lutInputs(cut.inputs.rbegin(), cut.inputs.rend());

    // Generate comb.truth table operation.
    auto truthTableOp = rewriter.create<comb::TruthTableOp>(
        cut.getRoot()->getLoc(), lutInputs, arrayAttr);

    // Replace the root operation with the truth table operation
    return truthTableOp.getOperation();
  }
};

struct GenericLUTMapperPass
    : public impl::GenericLutMapperBase<GenericLUTMapperPass> {
  using GenericLutMapperBase<GenericLUTMapperPass>::GenericLutMapperBase;
  void runOnOperation() override {
    // Add LUT pattern.
    auto *module = getOperation();
    SmallVector<std::unique_ptr<CutRewritePattern>> patterns;
    patterns.push_back(
        std::make_unique<GenericLUT>(module->getContext(), maxLutSize));
    CutRewritePatternSet patternSet(std::move(patterns));

    // Create the cut rewriter with the area optimization strategy.
    CutRewriterOptions options;
    // TODO: Currently we don't have implemented area-flow, so there is no
    //       difference in using area or timing.
    options.strategy = OptimizationStrategyTiming;
    options.maxCutInputSize = maxLutSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.attachDebugTiming = true; // Attach debug timing attributes
    CutRewriter mapper(options, patternSet);
    if (failed(mapper.run(module)))
      signalPassFailure();
  }
}; // namespace
} // namespace
