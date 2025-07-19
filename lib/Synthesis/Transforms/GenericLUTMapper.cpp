#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Synthesis/CutRewriter.h"
#include "circt/Synthesis/Transforms/Passes.h"

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

  DelayType getDelay(const Cut &cut, size_t inputIndex,
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

    // Reverse the inputs to match the LUT input order
    SmallVector<Value> lutInputs(cut.inputs.rbegin(), cut.inputs.rend());

    // Generate comb.truth table operation.
    auto truthTableOp = rewriter.create<comb::TruthTableOp>(
        cut.getRoot()->getLoc(), lutInputs, arrayAttr);

    // Replace the root operation with the truth table operation
    rewriter.replaceOp(cut.getRoot(), truthTableOp);
    return success();
  }
};
struct GenericLUTMapperPass
    : public impl::GenericLutMapperBase<GenericLUTMapperPass> {
  using GenericLutMapperBase<GenericLUTMapperPass>::GenericLutMapperBase;
  void runOnOperation() override {

    // Add LUT pattern.
    auto *module = getOperation();
    SmallVector<std::unique_ptr<CutRewriterPattern>> patterns;
    patterns.push_back(std::make_unique<GenericLUT>(maxLutSize));
    CutRewriterPatternSet patternSet(std::move(patterns));

    // Create the cut rewriter with the area optimization strategy.
    CutRewriterOptions options;
    // TODO: Currently we don't have implemented area-flow, so there is no
    //       difference in using area or timing.
    options.strategy = CutRewriteStrategy::Timing;
    options.maxCutInputSize = maxLutSize;
    options.maxCutSizePerRoot = maxCutsPerNode;
    CutRewriter mapper(options, patternSet);
    if (failed(mapper.run(module)))
      signalPassFailure();
  }
}; // namespace
} // namespace
