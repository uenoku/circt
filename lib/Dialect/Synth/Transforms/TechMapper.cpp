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

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
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

static llvm::FailureOr<BinaryTruthTable>
getCellTruthTable(hw::HWModuleOp module) {
  // Get input and output ports
  auto inputTypes = module.getInputTypes();
  auto outputTypes = module.getOutputTypes();

  unsigned numInputs = inputTypes.size();
  unsigned numOutputs = outputTypes.size();
  if (numOutputs != 1)
    return module->emitError(
        "Modules with multiple outputs are not supported yet");

  // Verify all ports are single bit
  for (auto type : inputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All input ports must be single bit");
  }
  for (auto type : outputTypes) {
    if (!type.isInteger(1))
      return module->emitError("All output ports must be single bit");
  }

  if (numInputs > maxTruthTableInputs)
    return module->emitError("Too many inputs for truth table generation");

  SmallVector<Value> results;
  results.reserve(numOutputs);
  // Get the body block of the module
  auto *bodyBlock = module.getBodyBlock();
  assert(bodyBlock && "Module must have a body block");
  // Collect output values from the body block
  for (auto result : bodyBlock->getTerminator()->getOperands())
    results.push_back(result);

  // Create a truth table for the module. Pin permutations are enumerated
  // by the pattern set at load time; no canonicalization is needed here.
  return getTruthTable(results, bodyBlock);
}

/// Simple technology library encoded as a HWModuleOp.
struct TechLibraryPattern : public CutRewritePattern {
  TechLibraryPattern(hw::HWModuleOp module, double area,
                     SmallVector<DelayType> delay, BinaryTruthTable table)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), module(module), table(std::move(table)) {

    LLVM_DEBUG({
      llvm::dbgs() << "Created Tech Library Pattern for module: "
                   << module.getModuleName() << "\n"
                   << "Table: " << this->table.table << "\n"
                   << "Inputs: " << this->table.numInputs << "\n";
    });
  }

  StringRef getPatternName() const override {
    auto moduleCp = module;
    return moduleCp.getModuleName();
  }

  /// Match the cut set against this library primitive. The caller
  /// guarantees N-canonical agreement (via the pattern-set lookup, for
  /// either output polarity); the binding describes how pattern pins
  /// map to the cut. Bindings with negation are phase-aware: each negated
  /// terminal materializes an inverter cell, with its area and delay
  /// folded into the match cost so selection stays timing- and
  /// area-accurate.
  std::optional<MatchResult> match(CutEnumerator &, const Cut &,
                                   const MatchBinding &binding) const override {
    if (!binding.hasNegation())
      return MatchResult(area, delay);
    if (!inverter || inverter->delay.empty())
      return std::nullopt;

    unsigned numInputs = binding.getNumInputs();
    unsigned numOutputs = numInputs ? delay.size() / numInputs : 0;
    double totalArea = area;
    SmallVector<DelayType, 6> adjusted(delay.begin(), delay.end());
    DelayType invDelay = inverter->delay.front();
    for (unsigned i = 0; i < numInputs; ++i) {
      if (binding.isPatternInputNegated(i)) {
        for (unsigned o = 0; o < numOutputs; ++o)
          adjusted[o * numInputs + i] += invDelay;
        totalArea += inverter->area;
      }
    }
    for (unsigned o = 0; o < numOutputs; ++o) {
      if (binding.isOutputNegated(o)) {
        for (unsigned i = 0; i < numInputs; ++i)
          adjusted[o * numInputs + i] += invDelay;
        totalArea += inverter->area;
      }
    }
    MatchResult result;
    result.area = totalArea;
    result.setOwnedDelays(std::move(adjusted));
    return result;
  }

  /// Enable truth table matching for this pattern
  bool useTruthTableMatcher(
      SmallVectorImpl<BinaryTruthTable> &matchingTables) const override {
    matchingTables.push_back(table);
    return true;
  }

  /// Rewrite the cut set using this library primitive
  llvm::FailureOr<Operation *>
  rewrite(mlir::OpBuilder &builder, CutEnumerator &enumerator, const Cut &cut,
          const MatchedPattern &matched) const override {
    const auto &network = enumerator.getLogicNetwork();
    const MatchBinding &binding = matched.getBinding();
    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");
    Location loc = rootOp->getLoc();
    // Negated pattern inputs observe inverted cut inputs through
    // inverter cells (guaranteed to exist: match() only accepts negated
    // bindings when an inverter pattern is available).
    assert((!binding.hasNegation() || inverter) &&
           "negated binding without inverter pattern");
    SmallVector<Value> inputs;
    unsigned numInputs = binding.getNumInputs();
    inputs.reserve(numInputs);
    for (unsigned patternInput = 0; patternInput < numInputs; ++patternInput) {
      unsigned idx = binding.getCutInputForPatternInput(patternInput);
      assert(idx < cut.inputs.size() && "input permutation index out of range");
      Value val = network.getValue(cut.inputs[idx]);
      if (binding.isPatternInputNegated(patternInput)) {
        auto invOp = hw::InstanceOp::create(builder, loc, inverter->module,
                                            "mapped_inv", ArrayRef<Value>(val));
        val = invOp.getResult(0);
      }
      inputs.push_back(val);
    }

    // TODO: Give a better name to the instance
    auto instanceOp = hw::InstanceOp::create(builder, loc, module, "mapped",
                                             ArrayRef<Value>(inputs));
    Operation *resultOp = instanceOp.getOperation();
    // A negated pattern output is inverted back for the cut consumers.
    // (Multi-output patterns are rejected upstream.)
    if (binding.isOutputNegated(0)) {
      auto invOp =
          hw::InstanceOp::create(builder, loc, inverter->module, "mapped_inv",
                                 ArrayRef<Value>(resultOp->getResult(0)));
      resultOp = invOp.getOperation();
    }
    return resultOp;
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

  double getArea() const { return area; }
  const BinaryTruthTable &getTruthTable() const { return table; }
  void setInverter(const TechLibraryPattern *inv) { inverter = inv; }

private:
  const double area;
  const SmallVector<DelayType> delay;
  hw::HWModuleOp module;
  BinaryTruthTable table;
  // Cheapest 1-input inverter cell, for phase-aware matches. Null when
  // the library has none (negated bindings then never match).
  const TechLibraryPattern *inverter = nullptr;
};

namespace {
struct TechMapperPass : public impl::TechMapperBase<TechMapperPass> {
  using TechMapperBase<TechMapperPass>::TechMapperBase;

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<std::unique_ptr<CutRewritePattern>> libraryPatterns;

    // Only cells with at most this many inputs become patterns (the reference
    // default cut size is 6; 7..9 supported via word-level transforms with
    // load-time permutation enumeration, 9-input functions via offline
    // witness tables). Wider cells are skipped instead of crashing the
    // canonicalizer. This also bounds cut enumeration.
    // Wider cells (e.g. 9-input AO333) use offline witness tables (see
    // the pattern-set constructor).
    constexpr unsigned kMaxPatternInputs = 9;
    // Consider modules with the "synth.mapping_cost" attribute as library
    // modules.
    SmallVector<hw::HWModuleOp> nonLibraryModules;
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {

      // Imported library leftovers (marked, without derived costs) and
      // dont_use cells are neither patterns nor design logic. Design
      // modules never carry these importer markers.
      if (!hwModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost") &&
          (hwModule->hasAttr("synth.liberty.cell") ||
           hwModule->hasAttr("synth.dont_use")))
        continue;

      // Sequential library cells imported from Liberty have executable
      // bodies built from synth.ff/synth.latch/synth.clock_gate. They hold
      // state, so they are neither combinational patterns nor mappable
      // design logic for cut rewriting; leave them alone.
      bool hasStateElement = hwModule.getBodyBlock()
                                 ->walk([](Operation *op) {
                                   if (isa<FFOp, LatchOp, ClockGateOp>(op))
                                     return WalkResult::interrupt();
                                   return WalkResult::advance();
                                 })
                                 .wasInterrupted();
      if (hasStateElement)
        continue;

      auto mappingCost =
          hwModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
      if (!mappingCost) {
        // Rejected supergate candidates (marked, without derived costs)
        // are neither patterns nor design logic; mapping them as design
        // would rewrite their bodies.
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

      auto arcs = mappingCost.getArcs();

      SmallVector<hw::PortInfo> inputPorts;
      for (const auto &port : hwModule.getPortList()) {
        if (!port.isInput())
          continue;
        inputPorts.push_back(port);
      }

      // Cells wider than the NPN cap never match capped cuts; skip
      // them instead of crashing the canonicalizer.
      if (inputPorts.size() > kMaxPatternInputs) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping wide cell (no NPN support): "
                                << hwModule.getModuleName() << "\n");
        continue;
      }

      if (arcs.size() != inputPorts.size()) {
        hwModule.emitError(
            "synth.mapping_cost arcs do not match module inputs");
        signalPassFailure();
        return;
      }

      SmallVector<DelayType> delay;
      for (auto attr : arcs) {
        auto arc = cast<LinearTimingArcAttr>(attr);

        // TechMapper currently preserves the old integer per-pin delay model.
        // The sensitivity, polarity, and input capacitance fields are carried
        // in the attribute for future load-aware mapping.
        delay.push_back(static_cast<DelayType>(arc.getIntrinsic()));
      }

      // Compute the raw truth table for the module. Pin permutations
      // are enumerated by the pattern set; no canonicalization here.
      auto table = getCellTruthTable(hwModule);
      if (failed(table)) {
        signalPassFailure();
        return;
      }

      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
          std::make_unique<TechLibraryPattern>(hwModule, area, std::move(delay),
                                               std::move(*table));

      // Add the pattern to the library
      libraryPatterns.push_back(std::move(pattern));
    }

    if (libraryPatterns.empty())
      return markAllAnalysesPreserved();

    // Phase-aware matching needs one inverter cell: the cheapest 1-input
    // pattern implementing negation (raw NOT table). Every pattern shares
    // it; without one, negated bindings keep failing as before.
    const TechLibraryPattern *inverter = nullptr;
    BinaryTruthTable notTable(1, 1, APInt(2, 1));
    for (auto &pattern : libraryPatterns) {
      auto *tech = static_cast<TechLibraryPattern *>(pattern.get());
      if (tech->getNumInputs() != 1 || tech->getNumOutputs() != 1)
        continue;
      if (!(tech->getTruthTable() == notTable))
        continue;
      if (!inverter || tech->getArea() < inverter->getArea())
        inverter = tech;
    }
    for (auto &pattern : libraryPatterns)
      static_cast<TechLibraryPattern *>(pattern.get())->setInverter(inverter);

    CutRewritePatternSet patternSet(std::move(libraryPatterns));
    CutRewriterOptions options;
    options.strategy = strategy;
    // Bound cut enumeration by the widest actually-matchable pattern:
    // skipped wide cells (no offline table) must not widen it.
    options.maxCutInputSize =
        std::min(patternSet.getMaxMatchedInputs(), kMaxPatternInputs);
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.attachDebugTiming = test;
    options.additionalCutRankings.push_back({compareCutsByAreaFlow, 1});
    options.additionalCutRankings.push_back({compareCutsByArea, 1});
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
};

} // namespace
