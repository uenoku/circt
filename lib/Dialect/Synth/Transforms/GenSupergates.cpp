//===- GenSupergates.cpp - Generate depth-2 supergate library -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates depth-2 supergates by composing pairs of primitive
// technology library cells. For each pair (innerCell, outerCell) and each input
// pin of outerCell, the inner cell's output is connected to that pin. The
// resulting composite cell is emitted as hw.module with hw.techlib.info.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_GENSUPERGATES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "synth-gen-supergates"

namespace {

struct CellInfo {
  hw::HWModuleOp module;
  double area;
  SmallVector<int64_t> delays; // per-input delay in ps
};

/// Collect all primitive cells (modules with hw.techlib.info).
static void collectBaseCells(ModuleOp topModule,
                             SmallVectorImpl<CellInfo> &cells) {
  for (auto hwModule : topModule.getOps<hw::HWModuleOp>()) {
    auto techInfo =
        hwModule->getAttrOfType<DictionaryAttr>("hw.techlib.info");
    if (!techInfo)
      continue;

    auto areaAttr = techInfo.getAs<FloatAttr>("area");
    auto delayAttr = techInfo.getAs<ArrayAttr>("delay");
    if (!areaAttr || !delayAttr)
      continue;

    CellInfo cell;
    cell.module = hwModule;
    cell.area = areaAttr.getValue().convertToDouble();
    for (auto delayValue : delayAttr) {
      auto delayArray = cast<ArrayAttr>(delayValue);
      for (auto delayElement : delayArray)
        cell.delays.push_back(
            cast<IntegerAttr>(delayElement).getValue().getZExtValue());
    }
    cells.push_back(std::move(cell));
  }
}

/// Compute NPN canonical form for an hw.module body.
static FailureOr<NPNClass> computeNPN(hw::HWModuleOp module) {
  auto outputTypes = module.getOutputTypes();
  if (outputTypes.size() != 1 || !outputTypes[0].isInteger(1))
    return failure();
  for (auto type : module.getInputTypes())
    if (!type.isInteger(1))
      return failure();
  if (module.getNumInputPorts() > synth::maxTruthTableInputs)
    return failure();

  auto *bodyBlock = module.getBodyBlock();
  SmallVector<Value> results;
  for (auto result : bodyBlock->getTerminator()->getOperands())
    results.push_back(result);

  auto truthTable = synth::getTruthTable(results, bodyBlock);
  if (failed(truthTable))
    return failure();

  return NPNClass::computeNPNCanonicalForm(*truthTable);
}

struct GenSupergatesPass
    : public circt::synth::impl::GenSupergatesBase<GenSupergatesPass> {
  using GenSupergatesBase::GenSupergatesBase;

  void runOnOperation() override {
    auto topModule = getOperation();
    auto *ctx = topModule.getContext();

    // Step 1: Collect base cells.
    SmallVector<CellInfo> baseCells;
    collectBaseCells(topModule, baseCells);
    if (baseCells.empty())
      return markAllAnalysesPreserved();

    // Step 2: Compute NPN classes for all base cells.
    DenseMap<llvm::APInt, double> coveredNPN; // NPN table -> best area
    for (auto &cell : baseCells) {
      auto npn = computeNPN(cell.module);
      if (failed(npn))
        continue;
      auto key = npn->truthTable.table;
      auto it = coveredNPN.find(key);
      if (it == coveredNPN.end() || cell.area < it->second)
        coveredNPN[key] = cell.area;
    }

    // Step 3: Generate supergates for each (inner, outer, pin) triple.
    unsigned supergateCount = 0;
    OpBuilder builder(ctx);
    // Insert before the first module's end.
    builder.setInsertionPointToEnd(topModule.getBody());

    struct SupergateCandidate {
      hw::HWModuleOp module;
      double area;
      SmallVector<int64_t> delays;
      llvm::APInt npnKey;
    };

    // Track best supergate per NPN class (only non-primitive ones).
    DenseMap<llvm::APInt, SupergateCandidate> bestSupergates;

    for (auto &inner : baseCells) {
      for (auto &outer : baseCells) {
        unsigned innerInputs = inner.module.getNumInputPorts();
        unsigned outerInputs = outer.module.getNumInputPorts();

        for (unsigned pin = 0; pin < outerInputs; ++pin) {
          // Total inputs = innerInputs + (outerInputs - 1).
          unsigned totalInputs = innerInputs + outerInputs - 1;
          if (totalInputs > maxInputs)
            continue;

          double totalArea = inner.area + outer.area;
          if (maxArea > 0.0 && totalArea > maxArea)
            continue;

          // Build the supergate module.
          SmallString<32> name;
          name = "__supergate_";
          name += std::to_string(supergateCount);

          // Build port list: innerInputs first, then outer inputs (skip pin).
          SmallVector<hw::PortInfo> ports;
          auto i1Ty = IntegerType::get(ctx, 1);

          unsigned portIdx = 0;
          // Inner cell inputs.
          for (unsigned i = 0; i < innerInputs; ++i) {
            auto inputName = inner.module.getInputName(i);
            ports.push_back({{StringAttr::get(ctx, inputName),
                              i1Ty,
                              hw::ModulePort::Direction::Input},
                             portIdx++});
          }
          // Outer cell inputs (skip the pin connected to inner output).
          for (unsigned i = 0; i < outerInputs; ++i) {
            if (i == pin)
              continue;
            auto inputName = outer.module.getInputName(i);
            // Disambiguate names by adding suffix if needed.
            SmallString<32> portName(inputName);
            portName += "_o";
            ports.push_back({{StringAttr::get(ctx, portName),
                              i1Ty,
                              hw::ModulePort::Direction::Input},
                             portIdx++});
          }
          // Output.
          ports.push_back({{StringAttr::get(ctx, "Y"),
                            i1Ty,
                            hw::ModulePort::Direction::Output},
                           portIdx});

          auto sgModule = hw::HWModuleOp::create(
              builder, builder.getUnknownLoc(),
              StringAttr::get(ctx, name), ports);
          sgModule.setPrivate();

          // Clone inner cell body ops into supergate body.
          auto *sgBody = sgModule.getBodyBlock();
          // Remove the auto-generated output op if present.
          if (sgBody->mightHaveTerminator())
            sgBody->getTerminator()->erase();

          IRMapping innerMapping;
          auto *innerBody = inner.module.getBodyBlock();
          // Map inner block args to supergate block args [0, innerInputs).
          for (unsigned i = 0; i < innerInputs; ++i)
            innerMapping.map(innerBody->getArgument(i),
                             sgBody->getArgument(i));

          builder.setInsertionPointToEnd(sgBody);
          // Clone inner body ops (except terminator).
          Value innerOutput;
          for (auto &op : innerBody->without_terminator()) {
            auto *cloned = builder.clone(op, innerMapping);
            // Track the result that feeds into inner's output.
            (void)cloned;
          }
          // The inner output is whatever the inner terminator references.
          auto *innerTerm = innerBody->getTerminator();
          innerOutput = innerMapping.lookup(innerTerm->getOperand(0));

          // Clone outer cell body ops.
          IRMapping outerMapping;
          auto *outerBody = outer.module.getBodyBlock();
          unsigned sgArgIdx = innerInputs;
          for (unsigned i = 0; i < outerInputs; ++i) {
            if (i == pin) {
              // Wire inner output to this pin.
              outerMapping.map(outerBody->getArgument(i), innerOutput);
            } else {
              outerMapping.map(outerBody->getArgument(i),
                               sgBody->getArgument(sgArgIdx++));
            }
          }

          for (auto &op : outerBody->without_terminator())
            builder.clone(op, outerMapping);

          // Create output op.
          auto *outerTerm = outerBody->getTerminator();
          Value outerOutput = outerMapping.lookup(outerTerm->getOperand(0));
          hw::OutputOp::create(builder, builder.getUnknownLoc(),
                               ValueRange{outerOutput});

          // Compute NPN class.
          auto npn = computeNPN(sgModule);
          if (failed(npn)) {
            sgModule.erase();
            continue;
          }

          auto npnKey = npn->truthTable.table;

          // Skip if already covered by a primitive cell.
          if (coveredNPN.count(npnKey)) {
            sgModule.erase();
            continue;
          }

          // Compute delays.
          SmallVector<int64_t> delays;
          // Inner inputs: delay = inner_delay[i] + outer_delay[pin].
          int64_t outerPinDelay =
              (pin < outer.delays.size()) ? outer.delays[pin] : 0;
          for (unsigned i = 0; i < innerInputs; ++i) {
            int64_t innerDelay =
                (i < inner.delays.size()) ? inner.delays[i] : 0;
            delays.push_back(innerDelay + outerPinDelay);
          }
          // Outer inputs (skip pin).
          for (unsigned i = 0; i < outerInputs; ++i) {
            if (i == pin)
              continue;
            int64_t d = (i < outer.delays.size()) ? outer.delays[i] : 0;
            delays.push_back(d);
          }

          // Check if this NPN class already has a better supergate.
          auto it = bestSupergates.find(npnKey);
          if (it != bestSupergates.end()) {
            if (totalArea >= it->second.area) {
              sgModule.erase();
              continue;
            }
            // This is better - erase the old one.
            it->second.module.erase();
            it->second = {sgModule, totalArea, std::move(delays), npnKey};
          } else {
            bestSupergates[npnKey] = {sgModule, totalArea, std::move(delays),
                                      npnKey};
          }

          supergateCount++;
        }
      }
    }

    // Step 4: Attach hw.techlib.info and synth.supergate to surviving modules.
    for (auto &[npnKey, candidate] : bestSupergates) {
      auto sgModule = candidate.module;
      // Rename to sequential.
      // Build delay attr.
      SmallVector<Attribute> delayPerInput;
      for (int64_t d : candidate.delays) {
        SmallVector<Attribute> inner;
        inner.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), d));
        delayPerInput.push_back(ArrayAttr::get(ctx, inner));
      }

      SmallVector<NamedAttribute> techInfoAttrs;
      techInfoAttrs.push_back(
          NamedAttribute(StringAttr::get(ctx, "area"),
                         FloatAttr::get(Float64Type::get(ctx), candidate.area)));
      techInfoAttrs.push_back(NamedAttribute(StringAttr::get(ctx, "delay"),
                                             ArrayAttr::get(ctx, delayPerInput)));

      sgModule->setAttr("hw.techlib.info",
                        DictionaryAttr::get(ctx, techInfoAttrs));
      sgModule->setAttr("synth.supergate",
                        BoolAttr::get(ctx, true));

      LLVM_DEBUG(llvm::dbgs() << "Generated supergate: "
                               << sgModule.getModuleName()
                               << " area=" << candidate.area << "\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "Generated " << bestSupergates.size()
                             << " supergates\n");
  }
};

} // namespace
