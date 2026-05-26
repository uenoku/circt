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
// "synth.mapping_cost" attributes as technology library patterns and maps
// non-library modules to optimal gate implementations based on area and timing
// optimization strategies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/TechLibraries.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
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

//===----------------------------------------------------------------------===//
// Tech Mapper Pass
//===----------------------------------------------------------------------===//

static llvm::FailureOr<NPNClass> getNPNClassFromModule(hw::HWModuleOp module) {
  FailureOr<BinaryTruthTable> truthTable = getTruthTable(module);
  if (failed(truthTable))
    return failure();

  return NPNClass::computeNPNCanonicalForm(*truthTable);
}

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<DelayType> delay, NPNClass npnClass,
                     hw::HWModuleOp inverterModule = {},
                     double inverterArea = 0.0, DelayType inverterDelay = 0)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), npnClass(std::move(npnClass)),
        inverterModule(inverterModule ? inverterModule.getOperation()
                                      : nullptr),
        inverterArea(inverterArea), inverterDelay(inverterDelay) {

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
    const auto &cutNPN = cut.getNPNClass(enumerator.getOptions().npnTable);
    if (!(cutNPN.truthTable == npnClass.truthTable))
      return std::nullopt;
    bool needsPhaseInverter = cutNPN.inputNegation != npnClass.inputNegation ||
                              cutNPN.outputNegation != npnClass.outputNegation;
    if (!inverterModule && needsPhaseInverter)
      return std::nullopt;
    if (!needsPhaseInverter)
      return MatchResult(area, delay);

    MatchResult result;
    result.area = area;
    SmallVector<DelayType, 6> phasedDelays(delay.begin(), delay.end());

    for (unsigned canonicalPos = 0, e = npnClass.inputPermutation.size();
         canonicalPos != e; ++canonicalPos) {
      unsigned patternInput = npnClass.inputPermutation[canonicalPos];
      bool invertInput = ((cutNPN.inputNegation >> canonicalPos) & 1) !=
                         ((npnClass.inputNegation >> canonicalPos) & 1);
      if (!invertInput)
        continue;
      result.area += inverterArea;
      for (unsigned output = 0, e = getNumOutputs(); output != e; ++output)
        phasedDelays[output * cut.getInputSize() + patternInput] +=
            inverterDelay;
    }

    if (cutNPN.outputNegation != npnClass.outputNegation) {
      result.area += inverterArea;
      for (DelayType &delay : phasedDelays)
        delay += inverterDelay;
    }

    result.setOwnedDelays(std::move(phasedDelays));
    return result;
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<NPNClass> &matchingNPNClasses) const override {
    matchingNPNClasses.push_back(npnClass);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *>
  rewrite(mlir::OpBuilder &builder, CutEnumerator &enumerator, const Cut &cut,
          const MatchedPattern &match) const override {
    const auto &network = enumerator.getLogicNetwork();
    // Create a new instance of the module
    ArrayRef<unsigned> permutedInputIndices = match.getInputMapping();
    ArrayRef<Phase> inputPhases = match.getInputPhases();

    SmallVector<Value> inputs;
    inputs.reserve(permutedInputIndices.size());
    for (unsigned idx : permutedInputIndices) {
      assert(idx < cut.inputs.size() && "input permutation index out of range");
      Value input = network.getValue(cut.inputs[idx]);
      if (inputPhases[idx] == Phase::Negative) {
        if (!inverterModule) {
          mlir::emitError(input.getLoc(),
                          "matched inverted input phase but no single-input "
                          "inverter is available");
          return failure();
        }
        SmallVector<Value, 1> invInputs{input};
        auto inv =
            hw::InstanceOp::create(builder, input.getLoc(), inverterModule,
                                   "phase_inv", ArrayRef<Value>(invInputs));
        input = inv.getResult(0);
      }
      inputs.push_back(input);
    }

    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");

    // TODO: Give a better name to the instance
    auto instanceOp = hw::InstanceOp::create(builder, rootOp->getLoc(), module,
                                             "mapped", ArrayRef<Value>(inputs));
    if (match.getResultPhase() == Phase::Negative) {
      if (!inverterModule) {
        rootOp->emitError("matched inverted output phase but no single-input "
                          "inverter is available");
        return failure();
      }
      SmallVector<Value, 1> invInputs{instanceOp.getResult(0)};
      auto inv =
          hw::InstanceOp::create(builder, rootOp->getLoc(), inverterModule,
                                 "phase_inv", ArrayRef<Value>(invInputs));
      return inv.getOperation();
    }
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
  Operation *inverterModule = nullptr;
  double inverterArea = 0.0;
  DelayType inverterDelay = 0;
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

    if (!builtinLibrary.empty() &&
        failed(appendBuiltinTechLibrary(module, builtinLibrary))) {
      signalPassFailure();
      return;
    }
    for (const auto &filename : externalLibraryFiles) {
      if (failed(appendTechLibraryFile(module, filename))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;
    hw::HWModuleOp inverterModule;
    double inverterArea = 0.0;
    DelayType inverterDelay = 0;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      if (!hwModule->hasAttr("synth.mapping_cost"))
        continue;
      if (hwModule.getNumInputPorts() != 1 || hwModule.getNumOutputPorts() != 1)
        continue;
      auto truthTable = getTruthTable(hwModule);
      if (failed(truthTable))
        continue;
      if (truthTable->table == APInt(2, 1)) {
        inverterModule = hwModule;
        break;
      }
    }
    if (inverterModule) {
      auto mappingCost =
          inverterModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      inverterArea = mappingCost.getArea().getValue().convertToDouble();
      for (auto attr : mappingCost.getArcs())
        inverterDelay = std::max(
            inverterDelay, static_cast<DelayType>(
                               cast<LinearTimingArcAttr>(attr).getIntrinsic()));
    }

    unsigned maxInputSize = 0;
    // Consider modules with the "synth.mapping_cost" attribute as library
    // modules.
    SmallVector<hw::HWModuleOp> nonLibraryModules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {

      auto mappingCost =
          hwModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      if (!mappingCost) {
        if (hwModule->hasAttr("synth.supergate_candidate") ||
            hwModule.getModuleName().starts_with("__supergate_"))
          continue;
        nonLibraryModules.push_back(hwModule);
        continue;
      }

      double area = mappingCost.getArea().getValue().convertToDouble();

      StringAttr outputName;
      hw::ModulePortInfo ports(hwModule.getPortList());
      for (const auto &port : ports.getOutputs()) {
        if (outputName) {
          hwModule.emitError(
              "Modules with multiple outputs are not supported yet");
          signalPassFailure();
          return;
        }
        outputName = port.name;
      }
      if (!outputName) {
        hwModule.emitError("expected library module to have an output");
        signalPassFailure();
        return;
      }

      llvm::DenseMap<StringAttr, DelayType> delayByInput;
      for (auto attr : mappingCost.getArcs()) {
        auto arc = cast<LinearTimingArcAttr>(attr);
        if (!arc) {
          hwModule.emitError(
              "expected synth.linear_timing_arc in synth.mapping_cost arcs");
          signalPassFailure();
          return;
        }

        if (arc.getPin() != outputName) {
          hwModule.emitError("mapping cost arc output '")
              << arc.getPin().getValue() << "' does not match module output '"
              << outputName.getValue() << "'";
          signalPassFailure();
          return;
        }

        int64_t intrinsicDelay = arc.getIntrinsic();

        // TechMapper currently preserves the old integer per-pin delay model.
        // The sensitivity, polarity, and input capacitance fields are carried
        // in the attribute for future load-aware mapping.
        if (!delayByInput
                 .try_emplace(arc.getRelatedPin(),
                              static_cast<DelayType>(intrinsicDelay))
                 .second) {
          hwModule.emitError("duplicate mapping cost arc for input '")
              << arc.getRelatedPin().getValue() << "'";
          signalPassFailure();
          return;
        }
      }

      SmallVector<DelayType> delay;
      for (const auto &port : hwModule.getPortList()) {
        if (!port.isInput())
          continue;

        auto it = delayByInput.find(port.name);
        if (it == delayByInput.end()) {
          hwModule.emitError("missing mapping cost arc for input '")
              << port.name.getValue() << "'";
          signalPassFailure();
          return;
        }

        delay.push_back(it->second);
      }

      if (delay.size() != delayByInput.size()) {
        hwModule.emitError(
            "synth.mapping_cost arcs do not match module inputs");
        signalPassFailure();
        return;
      }

      // Compute NPN Class for the module.
      auto npnClass = getNPNClassFromModule(hwModule);
      if (failed(npnClass)) {
        signalPassFailure();
        return;
      }

      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
          std::make_unique<TechLibraryPattern>(
              hwModule, area, std::move(delay), std::move(*npnClass),
              inverterModule, inverterArea, inverterDelay);

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
