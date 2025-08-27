//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test for priority cuts implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TESTPRIORITYCUTS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-test-priority-cuts"

//===----------------------------------------------------------------------===//
// Test Priority Cuts Pass
//===----------------------------------------------------------------------===//

namespace {

StringRef getTestVariableName(Value value) {
  if (auto *op = value.getDefiningOp()) {
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
      return name.getValue();
    return "<unknown>";
  }

  auto blockArg = cast<BlockArgument>(value);
  auto hwOp = dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
  if (!hwOp)
    return "<unknown>";
  return hwOp.getInputName(blockArg.getArgNumber());
}

struct DummyPattern : public CutRewritePattern {
  DummyPattern(mlir::MLIRContext *ctx) : CutRewritePattern(ctx) {}
  bool match(const Cut &cut) const override { return true; }
  FailureOr<Operation *> rewrite(mlir::OpBuilder &builder,
                                 Cut &cut) const override {
    return failure();
  }
  unsigned getNumInputs() const override { return 0; }
  unsigned getNumOutputs() const override { return 0; }
  double getArea(const Cut &cut) const override { return 0; }
  DelayType getDelay(unsigned inputIndex, unsigned outputIndex) const override {
    return 1;
  }
};

struct TestPriorityCutsPass
    : public impl::TestPriorityCutsBase<TestPriorityCutsPass> {
  using TestPriorityCutsBase<TestPriorityCutsPass>::TestPriorityCutsBase;

  void runOnOperation() override {
    auto hwModule = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running TestPriorityCuts on module: "
                            << hwModule.getName() << "\n");

    // Set up cut rewriter options for testing
    CutRewriterOptions options;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    // Currently there is no behavioral difference between timing and area
    // so just test with timing strategy.
    options.strategy = circt::synth::OptimizationStrategyTiming;
    options.testPriorityCuts = true;
    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    patterns.push_back(std::make_unique<DummyPattern>(hwModule->getContext()));
    CutRewritePatternSet patternSet(std::move(patterns));
    CutRewriter rewriter(options, patternSet);
    CutEnumerator enumerator(options);
    llvm::outs() << "Enumerating cuts for module: " << hwModule.getModuleName()
                 << "\n";

    if (failed(rewriter.run(hwModule))) {
      signalPassFailure();
      return;
    }

    llvm::outs() << "Cut enumeration completed successfully\n";

    // This is a test pass, so we don't modify the IR
    markAllAnalysesPreserved();
  }
};

} // namespace
