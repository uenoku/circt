//===- GenSupergates.cpp - Generate supergate library ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Dialect/Synth/Transforms/TechLibraries.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_GENSUPERGATES
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-gen-supergates"

namespace {

struct CellInfo {
  hw::HWModuleOp module;
  double area;
  SmallVector<int64_t> delays;
  int64_t maxDelay;
  unsigned numGates;
};

static FailureOr<CellInfo> getCellInfo(hw::HWModuleOp hwModule,
                                       MappingCostAttr mappingCost) {
  double area = mappingCost.getArea().getValue().convertToDouble();

  StringAttr outputName;
  hw::ModulePortInfo ports(hwModule.getPortList());
  for (const auto &port : ports.getOutputs()) {
    if (outputName)
      return hwModule.emitError(
          "Modules with multiple outputs are not supported yet");
    outputName = port.name;
  }
  if (!outputName)
    return hwModule.emitError("expected library module to have an output");

  llvm::DenseMap<StringAttr, int64_t> delayByInput;
  for (auto attr : mappingCost.getArcs()) {
    auto arc = dyn_cast<LinearTimingArcAttr>(attr);
    if (!arc)
      return hwModule.emitError(
          "expected synth.linear_timing_arc in synth.mapping_cost arcs");
    if (arc.getPin() != outputName)
      return hwModule.emitError("mapping cost arc output '")
             << arc.getPin().getValue() << "' does not match module output '"
             << outputName.getValue() << "'";
    if (!delayByInput.try_emplace(arc.getRelatedPin(), arc.getIntrinsic())
             .second)
      return hwModule.emitError("duplicate mapping cost arc for input '")
             << arc.getRelatedPin().getValue() << "'";
  }

  CellInfo info;
  info.module = hwModule;
  info.area = area;
  info.numGates = 1;
  for (const auto &port : hwModule.getPortList()) {
    if (!port.isInput())
      continue;
    auto it = delayByInput.find(port.name);
    if (it == delayByInput.end())
      return hwModule.emitError("missing mapping cost arc for input '")
             << port.name.getValue() << "'";
    info.delays.push_back(it->second);
  }

  if (info.delays.size() != delayByInput.size())
    return hwModule.emitError("synth.mapping_cost arcs do not match module "
                              "inputs");
  if (info.delays.empty())
    info.maxDelay = 0;
  else
    info.maxDelay = *llvm::max_element(info.delays);
  return info;
}

static LogicalResult collectBaseCells(ModuleOp topModule,
                                      SmallVectorImpl<CellInfo> &cells) {
  for (auto hwModule : topModule.getOps<hw::HWModuleOp>()) {
    if (hwModule->hasAttr("synth.supergate"))
      continue;

    auto mappingCost =
        hwModule->getAttrOfType<MappingCostAttr>("synth.mapping_cost");
    if (!mappingCost)
      continue;

    auto info = getCellInfo(hwModule, mappingCost);
    if (failed(info))
      return failure();
    cells.push_back(std::move(*info));
  }
  return success();
}

static FailureOr<NPNClass> computeNPN(hw::HWModuleOp module) {
  auto tt = synth::getTruthTable(module);
  if (failed(tt))
    return failure();
  return NPNClass::computeNPNCanonicalForm(*tt);
}

using NPNKey = std::pair<llvm::APInt, uint64_t>;

static NPNKey makeNPNKey(const NPNClass &npn) {
  uint64_t negKey =
      (static_cast<uint64_t>(npn.inputNegation) << 32) | npn.outputNegation;
  return {npn.truthTable.table, negKey};
}

static hw::HWModuleOp createSupergateModule(
    OpBuilder &builder, Location loc, StringAttr name, CellInfo &inner,
    CellInfo &outer, unsigned pin, ArrayRef<unsigned> innerToInput,
    ArrayRef<unsigned> outerBypassToInput, unsigned totalInputs) {
  auto savedInsertion = builder.saveInsertionPoint();
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;

  for (unsigned i = 0; i < totalInputs; ++i) {
    SmallString<16> inName("in");
    inName += std::to_string(i);
    ports.push_back(
        {{StringAttr::get(ctx, inName), i1, hw::ModulePort::Direction::Input},
         i});
  }

  ports.push_back(
      {{StringAttr::get(ctx, "Y"), i1, hw::ModulePort::Direction::Output},
       totalInputs});

  auto sg = hw::HWModuleOp::create(builder, loc, name, ports);
  sg.setPrivate();

  auto *body = sg.getBodyBlock();
  if (body->mightHaveTerminator())
    body->getTerminator()->erase();

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> innerOperands;
  for (unsigned idx : innerToInput)
    innerOperands.push_back(body->getArgument(idx));

  auto innerInst = hw::InstanceOp::create(builder, loc, inner.module, "inner",
                                          innerOperands);

  SmallVector<Value> outerOperands;
  unsigned bypassIdx = 0;
  for (unsigned i = 0, e = outer.module.getNumInputPorts(); i != e; ++i) {
    if (i == pin) {
      outerOperands.push_back(innerInst.getResult(0));
      continue;
    }
    outerOperands.push_back(body->getArgument(outerBypassToInput[bypassIdx++]));
  }

  auto outerInst = hw::InstanceOp::create(builder, loc, outer.module, "outer",
                                          outerOperands);
  hw::OutputOp::create(builder, loc, ValueRange{outerInst.getResult(0)});
  builder.restoreInsertionPoint(savedInsertion);
  return sg;
}

static MappingCostAttr buildMappingCostAttr(MLIRContext *ctx,
                                            hw::HWModuleOp module, double area,
                                            ArrayRef<int64_t> delays) {
  StringAttr outputName;
  for (const auto &port : module.getPortList())
    if (port.isOutput()) {
      outputName = port.name;
      break;
    }

  auto polarity = PolarityAttr::get(ctx, PolarityKind::Positive);
  SmallVector<Attribute> arcs;
  unsigned inputIdx = 0;
  for (const auto &port : module.getPortList()) {
    if (!port.isInput())
      continue;
    arcs.push_back(LinearTimingArcAttr::get(ctx, outputName, port.name,
                                            delays[inputIdx++],
                                            /*sensitivity=*/0, polarity));
  }

  return MappingCostAttr::get(ctx, FloatAttr::get(Float64Type::get(ctx), area),
                              ArrayAttr::get(ctx, arcs),
                              DictionaryAttr::get(ctx));
}

struct SupergateCandidate {
  hw::HWModuleOp module;
  double area;
  SmallVector<int64_t> delays;
  int64_t maxDelay;
  unsigned numGates;
};

struct GenSupergatesPass
    : public circt::synth::impl::GenSupergatesBase<GenSupergatesPass> {
  using GenSupergatesBase::GenSupergatesBase;

  void runOnOperation() override {
    auto topModule = getOperation();
    auto *ctx = topModule.getContext();

    if (!builtinLibrary.empty() &&
        failed(appendBuiltinTechLibrary(topModule, builtinLibrary))) {
      signalPassFailure();
      return;
    }
    for (const auto &filename : externalLibraryFiles) {
      if (failed(appendTechLibraryFile(topModule, filename))) {
        signalPassFailure();
        return;
      }
    }

    if (maxGates < 2)
      return;

    SmallVector<CellInfo> baseCells;
    if (failed(collectBaseCells(topModule, baseCells))) {
      signalPassFailure();
      return;
    }
    if (baseCells.empty())
      return;

    DenseMap<NPNKey, double> coveredNPN;
    for (auto &cell : baseCells) {
      auto npn = computeNPN(cell.module);
      if (failed(npn)) {
        signalPassFailure();
        return;
      }
      auto key = makeNPNKey(*npn);
      auto it = coveredNPN.find(key);
      if (it == coveredNPN.end() || cell.area < it->second)
        coveredNPN[key] = cell.area;
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(topModule.getBody());
    Location loc = topModule.getLoc();

    DenseMap<NPNKey, SupergateCandidate> bestByNPN;
    unsigned supergateOrdinal = 0;

    SmallVector<CellInfo> frontier = baseCells;
    for (unsigned depth = 2; depth <= maxGates && !frontier.empty(); ++depth) {
      std::sort(frontier.begin(), frontier.end(),
                [](const CellInfo &a, const CellInfo &b) {
                  if (a.maxDelay != b.maxDelay)
                    return a.maxDelay < b.maxDelay;
                  return a.area < b.area;
                });

      for (auto &outer : baseCells) {
        unsigned outerInputs = outer.module.getNumInputPorts();
        SmallVector<int64_t> outerMaxExcludingPin(outerInputs, 0);
        for (unsigned pin = 0; pin < outerInputs; ++pin) {
          int64_t maxBypass = 0;
          for (unsigned i = 0; i < outerInputs; ++i)
            if (i != pin)
              maxBypass = std::max(maxBypass, outer.delays[i]);
          outerMaxExcludingPin[pin] = maxBypass;
        }

        for (unsigned pin = 0; pin < outerInputs; ++pin) {
          unsigned triedForRootPin = 0;
          int64_t maxInnerDelayBudget =
              maxDelay == 0 ? std::numeric_limits<int64_t>::max()
                            : maxDelay - outer.delays[pin];

          for (auto &inner : frontier) {
            if (inner.maxDelay > maxInnerDelayBudget)
              break;
            if (maxCandidatesPerRoot > 0 &&
                triedForRootPin >= maxCandidatesPerRoot)
              break;

            unsigned innerInputs = inner.module.getNumInputPorts();
            double area = inner.area + outer.area;
            if (maxArea > 0.0 && area > maxArea)
              continue;

            int64_t candidateMaxDelay = std::max(
                inner.maxDelay + outer.delays[pin], outerMaxExcludingPin[pin]);
            if (maxDelay > 0 && candidateMaxDelay > maxDelay)
              continue;

            auto tryCandidate = [&](ArrayRef<unsigned> innerToInput,
                                    ArrayRef<unsigned> outerBypassToInput,
                                    unsigned totalInputs) {
              if (totalInputs > maxInputs)
                return;

              ++triedForRootPin;
              builder.setInsertionPointToEnd(topModule.getBody());
              SmallString<32> supergateName("__supergate_");
              supergateName += std::to_string(supergateOrdinal++);
              auto sg = createSupergateModule(
                  builder, loc, StringAttr::get(ctx, supergateName), inner,
                  outer, pin, innerToInput, outerBypassToInput, totalInputs);

              auto npn = computeNPN(sg);
              if (failed(npn))
                return;

              auto key = makeNPNKey(*npn);
              if (coveredNPN.count(key))
                return;

              SmallVector<int64_t> delays(totalInputs, 0);
              int64_t viaInner = outer.delays[pin];
              for (unsigned i = 0; i < innerInputs; ++i) {
                unsigned inputIdx = innerToInput[i];
                delays[inputIdx] =
                    std::max(delays[inputIdx], inner.delays[i] + viaInner);
              }

              unsigned bypassIdx = 0;
              for (unsigned i = 0; i < outerInputs; ++i) {
                if (i == pin)
                  continue;
                unsigned inputIdx = outerBypassToInput[bypassIdx++];
                delays[inputIdx] = std::max(delays[inputIdx], outer.delays[i]);
              }

              auto it = bestByNPN.find(key);
              if (it == bestByNPN.end() || area < it->second.area)
                bestByNPN[key] = {sg, area, std::move(delays),
                                  candidateMaxDelay, depth};
            };

            unsigned baseTotalInputs = innerInputs + outerInputs - 1;
            SmallVector<unsigned> baseInnerToInput;
            SmallVector<unsigned> baseOuterBypassToInput;
            baseInnerToInput.reserve(innerInputs);
            baseOuterBypassToInput.reserve(outerInputs > 0 ? outerInputs - 1
                                                           : 0);

            for (unsigned i = 0; i < innerInputs; ++i)
              baseInnerToInput.push_back(i);
            unsigned nextInput = innerInputs;
            for (unsigned i = 0; i < outerInputs; ++i) {
              if (i == pin)
                continue;
              baseOuterBypassToInput.push_back(nextInput++);
            }
            tryCandidate(baseInnerToInput, baseOuterBypassToInput,
                         baseTotalInputs);

            if (!this->allowDuplicateInputs || baseTotalInputs <= 1)
              continue;

            for (unsigned innerIdx = 0; innerIdx < innerInputs; ++innerIdx) {
              unsigned bypassPos = 0;
              for (unsigned outerInput = 0; outerInput < outerInputs;
                   ++outerInput) {
                if (outerInput == pin)
                  continue;

                SmallVector<unsigned> dupInnerToInput(baseInnerToInput);
                SmallVector<unsigned> dupOuterBypassToInput(
                    baseOuterBypassToInput);

                unsigned mergedInput = dupInnerToInput[innerIdx];
                unsigned removedInput = dupOuterBypassToInput[bypassPos];
                dupOuterBypassToInput[bypassPos] = mergedInput;

                for (auto &idx : dupOuterBypassToInput)
                  if (idx > removedInput)
                    --idx;

                tryCandidate(dupInnerToInput, dupOuterBypassToInput,
                             baseTotalInputs - 1);
                ++bypassPos;
              }
            }

            if (maxCandidatesPerRoot > 0 &&
                triedForRootPin >= maxCandidatesPerRoot)
              break;
          }
        }
      }

      SmallVector<CellInfo> nextFrontier;
      for (auto &entry : bestByNPN) {
        auto &candidate = entry.second;
        if (candidate.numGates != depth)
          continue;
        nextFrontier.push_back({candidate.module, candidate.area,
                                candidate.delays, candidate.maxDelay,
                                candidate.numGates});
      }
      frontier = std::move(nextFrontier);
    }

    for (auto &entry : bestByNPN) {
      auto &candidate = entry.second;
      candidate.module->setAttr("synth.mapping_cost",
                                buildMappingCostAttr(ctx, candidate.module,
                                                     candidate.area,
                                                     candidate.delays));
      candidate.module->setAttr("synth.supergate", BoolAttr::get(ctx, true));

      LLVM_DEBUG(llvm::dbgs() << "Generated supergate: "
                              << candidate.module.getModuleName() << "\n");
    }

    llvm::SmallPtrSet<Operation *, 16> liveSupergates;
    for (auto &entry : bestByNPN)
      liveSupergates.insert(entry.second.module.getOperation());

    llvm::SmallPtrSet<Operation *, 16> erasedModules;
    for (auto module :
         llvm::make_early_inc_range(topModule.getOps<hw::HWModuleOp>())) {
      if (!module.getModuleName().starts_with("__supergate_"))
        continue;
      Operation *op = module.getOperation();
      if (liveSupergates.contains(op) || !erasedModules.insert(op).second)
        continue;
      module.erase();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Generated " << bestByNPN.size() << " supergates\n");
  }
};

} // namespace
