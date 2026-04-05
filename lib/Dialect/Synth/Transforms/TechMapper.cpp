//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the TechMapper pass, which performs technology mapping
// by converting logic network representations (AIG operations) into
// technology-specific gate implementations using cut-based rewriting.
//
// The pass uses a cut-based algorithm with priority cuts and NPN canonical
// forms for efficient pattern matching. It processes HWModuleOp instances with
// "hw.techlib.info" attributes as technology library patterns and maps
// non-library modules to optimal gate implementations based on area and timing
// optimization strategies.
//
//===----------------------------------------------------------------------===//

#include "CutRewriteDBImpl.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <atomic>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TECHMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-tech-mapper"

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<DelayType> delay, NPNClass npnClass)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), npnClass(std::move(npnClass)) {

    LLVM_DEBUG({
      llvm::dbgs() << "Created Tech Library Pattern for module: "
                   << module.getModuleName() << "\n"
                   << "NPN Class: " << this->npnClass.truthTable.table << "\n"
                   << "Inputs: " << this->npnClass.inputPermutation.size()
                   << "\n"
                   << "Input Negation: " << this->npnClass.inputNegation << "\n"
                   << "Output Negation: " << this->npnClass.outputNegation
                   << "\n";
    });
  }

  StringRef getPatternName() const override {
    auto moduleCp = module;
    return moduleCp.getModuleName();
  }

  /// Match the cut set against this library primitive
  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    if (!cut.getNPNClass(enumerator.getOptions())
             .equivalentOtherThanPermutation(npnClass))
      return std::nullopt;

    return MatchResult(area, delay);
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &builder,
                                       CutEnumerator &enumerator,
                                       const Cut &cut,
                                       const MatchedPattern &match) const override {
    const auto &network = enumerator.getLogicNetwork();
    // Create a new instance of the module
    SmallVector<Value> inputs;
    const auto &binding = match.getBinding();
    inputs.reserve(binding.inputPermutation.size());
    for (unsigned idx : binding.inputPermutation) {
      assert(idx < cut.inputs.size() && "input permutation index out of range");
      inputs.push_back(network.getValue(cut.inputs[idx]));
    }

    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");

    // TODO: Give a better name to the instance
    auto instanceOp = hw::InstanceOp::create(builder, rootOp->getLoc(), module,
                                             "mapped", ArrayRef<Value>(inputs));
    return instanceOp.getOperation();
  }

  unsigned getNumInputs() const {
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
  const SmallVector<DelayType> delay;
  hw::HWModuleOp module;
  NPNClass npnClass;
};

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  LogicalResult initialize(MLIRContext *context) override {
    (void)context;
    npnTable = std::make_shared<const NPNTable>();
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;

    unsigned maxInputSize = 0;
    // Consider modules with the "hw.techlib.info" attribute as library
    // modules.
    // TODO: This attribute should be replaced with a more structured
    // representation of technology library information. Specifically, we should
    // have a dedicated operation for technology library.
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

      auto areaAndDelay = getAreaAndDelayFromTechInfo(hwModule);
      if (failed(areaAndDelay)) {
        signalPassFailure();
        return;
      }
      double area = areaAndDelay->first;
      SmallVector<DelayType> delay = std::move(areaAndDelay->second);
      // Compute NPN Class for the module.
      auto npnClass = getNPNClassFromModule(hwModule);
      if (failed(npnClass)) {
        signalPassFailure();
        return;
      }

      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
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
    options.npnTable = npnTable.get();
    std::atomic<uint64_t> numCutsCreatedCount = 0;
    std::atomic<uint64_t> numCutSetsCreatedCount = 0;
    std::atomic<uint64_t> numCutsRewrittenCount = 0;
    auto result = mlir::failableParallelForEach(
        module.getContext(), nonLibraryModules, [&](hw::HWModuleOp hwModule) {
          LLVM_DEBUG(llvm::dbgs() << "Processing non-library module: "
                                  << hwModule.getName() << "\n");
          CutRewriter rewriter(options, patternSet);
          if (failed(rewriter.run(hwModule)))
            return failure();
          const auto &stats = rewriter.getStats();
          numCutsCreatedCount.fetch_add(stats.numCutsCreated,
                                        std::memory_order_relaxed);
          numCutSetsCreatedCount.fetch_add(stats.numCutSetsCreated,
                                           std::memory_order_relaxed);
          numCutsRewrittenCount.fetch_add(stats.numCutsRewritten,
                                          std::memory_order_relaxed);
          return success();
        });
    if (failed(result))
      signalPassFailure();
    numCutsCreated += numCutsCreatedCount;
    numCutSetsCreated += numCutSetsCreatedCount;
    numCutsRewritten += numCutsRewrittenCount;
  }

private:
  std::shared_ptr<const NPNTable> npnTable;
};

} // namespace
