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
#include <algorithm>
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
                     SmallVector<DelayType> delay,
                     SmallVector<int64_t> sensitivity, double meanInputCap,
                     BinaryTruthTable table)
      : CutRewritePattern(module->getContext()), area(area),
        delay(std::move(delay)), sensitivity(std::move(sensitivity)),
        meanInputCap(meanInputCap), module(module), table(std::move(table)) {

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
  /// area-accurate. Delays are load-aware (Stage 1): each arc reports
  /// intrinsic + sensitivity x output_load, with output_load estimated
  /// from the cut root's subject-graph fanout times the library mean
  /// input capacitance (no cover context needed).
  std::optional<MatchResult> match(CutEnumerator &enumerator, const Cut &cut,
                                   const MatchBinding &binding) const override {
    unsigned numInputs = binding.getNumInputs();
    unsigned numOutputs = numInputs ? delay.size() / numInputs : 0;
    // Fast path: load-independent arcs without negation report intrinsics
    // directly (borrowed, as before).
    if (!binding.hasNegation() &&
        llvm::all_of(sensitivity, [](int64_t s) { return s == 0; }))
      return MatchResult(area, delay);
    if (binding.hasNegation() && (!inverter || inverter->delay.empty()))
      return std::nullopt;
    // Stage-1 load estimate: subject-graph fanout x mean input cap.
    // Deterministic: refcount and mean cap are both order-independent.
    // The fanout is clamped: broadcast nets above the cap are assumed to
    // be buffered downstream (buffer insertion), so mapping must not see
    // linear-model loads extrapolated orders of magnitude past the fitted
    // NLDM range (which otherwise dominates and distorts selection).
    // Trusted-fanout gate: the subject-fanout proxy is only applied to
    // small-fanout roots, where subject fanout still resembles cover
    // load. On heavily shared/broadcast roots the proxy lies (cover
    // sharing and duplication reshape fanout), and arrival-only
    // penalties without criticality context systematically mislead
    // selection and arrival-ordered pruning; those roots stay
    // load-blind (intrinsic delays).
    constexpr unsigned kMaxFanoutForLoadEstimate = 64;
    constexpr unsigned kTrustedFanoutForLoadEstimate = 8;
    unsigned fanout =
        enumerator.getLogicNetwork().getTotalRefCount(cut.getRootIndex());
    double load =
        fanout > kTrustedFanoutForLoadEstimate
            ? 0.0
            : meanInputCap * std::min(fanout, kMaxFanoutForLoadEstimate);
    SmallVector<DelayType, 6> adjusted;
    adjusted.reserve(delay.size());
    // The load penalty per arc is capped at one intrinsic: broadcast
    // loads are a buffering problem (buffer insertion), not a selection
    // signal, and uncapped penalties let non-critical broadcast nets
    // dominate selection and arrival-ordered pruning. Normal-fanout
    // penalties (far below intrinsic) are unaffected.
    for (unsigned o = 0; o < numOutputs; ++o)
      for (unsigned i = 0; i < numInputs; ++i) {
        DelayType intrinsic = delay[o * numInputs + i];
        DelayType penalty =
            std::min<DelayType>(llround(sensitivity[i] * load), intrinsic);
        adjusted.push_back(intrinsic + penalty);
      }
    double totalArea = area;
    if (binding.hasNegation()) {
      // Input-negating inverters drive one cell input (mean-cap load);
      // an output-negating inverter drives this cell's output load.
      // Same per-arc penalty cap as above.
      DelayType inInvDelay =
          inverter->delay.front() +
          std::min<DelayType>(
              llround(inverter->sensitivity.front() * meanInputCap),
              inverter->delay.front());
      DelayType outInvDelay =
          inverter->delay.front() +
          std::min<DelayType>(llround(inverter->sensitivity.front() * load),
                              inverter->delay.front());
      for (unsigned i = 0; i < numInputs; ++i) {
        if (binding.isPatternInputNegated(i)) {
          for (unsigned o = 0; o < numOutputs; ++o)
            adjusted[o * numInputs + i] += inInvDelay;
          totalArea += inverter->area;
        }
      }
      for (unsigned o = 0; o < numOutputs; ++o) {
        if (binding.isOutputNegated(o)) {
          for (unsigned i = 0; i < numInputs; ++i)
            adjusted[o * numInputs + i] += outInvDelay;
          totalArea += inverter->area;
        }
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
  // Per-pattern-input load sensitivities (ps per load unit), parallel to
  // the input dimension of `delay`; plus the library mean input cap (pF)
  // for Stage-1 load estimates.
  const SmallVector<int64_t> sensitivity;
  const double meanInputCap;
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
    // modules. Priced cells are collected first so the library mean input
    // capacitance (Stage-1 load estimates) is known before patterns are
    // built; module/port order keeps it deterministic.
    struct PricedCell {
      hw::HWModuleOp module;
      double area;
      SmallVector<DelayType> delay;
      SmallVector<int64_t> sensitivity;
    };
    SmallVector<PricedCell> pricedCells;
    double capSum = 0.0;
    size_t capCount = 0;
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
      SmallVector<int64_t> sensitivity;
      for (auto attr : arcs) {
        auto arc = cast<LinearTimingArcAttr>(attr);
        delay.push_back(static_cast<DelayType>(arc.getIntrinsic()));
        sensitivity.push_back(arc.getSensitivity());
      }
      // Accumulate input capacitances for the library mean (pF).
      if (auto inputCaps = mappingCost.getInputCaps()) {
        for (const auto &port : inputPorts) {
          if (auto cap = dyn_cast_or_null<FloatAttr>(inputCaps.get(port.name)))
            capSum += cap.getValue().convertToDouble(), ++capCount;
        }
      }
      pricedCells.push_back(
          {hwModule, area, std::move(delay), std::move(sensitivity)});
    }

    double meanInputCap = capCount == 0 ? 0.0 : capSum / capCount;
    for (auto &cell : pricedCells) {
      // Compute the raw truth table for the module. Pin permutations
      // are enumerated by the pattern set; no canonicalization here.
      auto table = getCellTruthTable(cell.module);
      if (failed(table)) {
        signalPassFailure();
        return;
      }
      // Load-aware matching is opt-in; when off, zero sensitivities take
      // the borrowed-intrinsics fast path in match() (bit-identical to
      // the old load-blind behavior).
      SmallVector<int64_t> sens =
          loadAwareDelay ? std::move(cell.sensitivity)
                         : SmallVector<int64_t>(cell.sensitivity.size(), 0);
      // Create a CutRewritePattern for the library module
      std::unique_ptr<TechLibraryPattern> pattern =
          std::make_unique<TechLibraryPattern>(
              cell.module, cell.area, std::move(cell.delay), std::move(sens),
              meanInputCap, std::move(*table));

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

    CutRewritePatternSet patternSet(std::move(libraryPatterns),
                                    maxEntriesPerKey);
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
    if (reserveIngredientCuts > 0)
      options.additionalCutRankings.push_back(
          {compareCutsBySmallInputs, reserveIngredientCuts});
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
