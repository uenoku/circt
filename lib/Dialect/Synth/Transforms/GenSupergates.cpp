//===- GenSupergates.cpp - Generate supergate library ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates supergates by composing primitive technology library
// cells (plus optional inverters for phase-flipped variants). Supergates are
// emitted as hw.modules with synth.mapping_cost so TechMapper picks them up
// transparently as ordinary patterns.
//
// Only the best-area supergate per exact function is kept, and functions
// already covered by primitive cells are skipped. Pin diversity across
// same-function variants is preserved downstream by load-time permutation
// enumeration in the pattern set.
//
// Adapted from the phase-aware mapping prototype (ueno/dev/hidetou/tech-lib)
// to N-canonical matching: dedup and coverage keys are raw truth tables
// instead of NPN classes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>
#include <optional>
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
  BinaryTruthTable table;
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

  auto arcs = mappingCost.getArcs();
  if (arcs.size() != hwModule.getNumInputPorts())
    return hwModule.emitError("synth.mapping_cost arcs do not match module "
                              "inputs");

  CellInfo info;
  info.module = hwModule;
  info.area = area;
  info.numGates = 1;
  for (auto attr : arcs) {
    auto arc = dyn_cast<LinearTimingArcAttr>(attr);
    if (!arc)
      return hwModule.emitError(
          "expected synth.linear_timing_arc in synth.mapping_cost arcs");
    info.delays.push_back(static_cast<int64_t>(arc.getIntrinsic()));
  }

  if (info.delays.empty())
    info.maxDelay = 0;
  else
    info.maxDelay = *llvm::max_element(info.delays);
  auto table = synth::getTruthTable(hwModule);
  if (failed(table))
    return failure();
  info.table = std::move(*table);
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

static std::optional<CellInfo> findCheapestInverter(ArrayRef<CellInfo> cells) {
  std::optional<CellInfo> best;
  for (auto cell : cells) {
    if (cell.module.getNumInputPorts() != 1 ||
        cell.module.getNumOutputPorts() != 1)
      continue;
    auto tt = synth::getTruthTable(cell.module);
    if (failed(tt) || tt->table != llvm::APInt(2, 1))
      continue;
    if (!best || cell.area < best->area ||
        (cell.area == best->area && cell.maxDelay < best->maxDelay))
      best = cell;
  }
  return best;
}

/// Raw-table key for supergate dedup and coverage: exact function plus input
/// count (bit width alone is ambiguous across arities).
using RawTableKey = std::pair<llvm::APInt, unsigned>;

static RawTableKey makeRawKey(const BinaryTruthTable &tt) {
  return {tt.table, tt.numInputs};
}

static hw::HWModuleOp createSupergateModule(
    OpBuilder &builder, Location loc, StringAttr name, CellInfo &inner,
    CellInfo &outer, unsigned pin, ArrayRef<unsigned> innerToInput,
    ArrayRef<unsigned> outerBypassToInput, unsigned totalInputs,
    const CellInfo *innerInverter = nullptr) {
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
  sg->setAttr("synth.supergate_candidate", BoolAttr::get(ctx, true));

  auto *body = sg.getBodyBlock();
  if (body->mightHaveTerminator())
    body->getTerminator()->erase();

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> innerOperands;
  for (unsigned idx : innerToInput)
    innerOperands.push_back(body->getArgument(idx));

  auto innerInst = hw::InstanceOp::create(builder, loc, inner.module, "inner",
                                          innerOperands);
  Value innerResult = innerInst.getResult(0);
  if (innerInverter) {
    SmallVector<Value, 1> invOperands{innerResult};
    auto invInst =
        hw::InstanceOp::create(builder, loc, innerInverter->module, "inner_inv",
                               ArrayRef<Value>(invOperands));
    innerResult = invInst.getResult(0);
  }

  SmallVector<Value> outerOperands;
  unsigned bypassIdx = 0;
  for (unsigned i = 0, e = outer.module.getNumInputPorts(); i != e; ++i) {
    if (i == pin) {
      outerOperands.push_back(innerResult);
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
  auto polarity = PolarityAttr::get(ctx, PolarityKind::Positive);
  SmallVector<Attribute> arcs;
  for (const auto &port : module.getPortList()) {
    if (!port.isInput())
      continue;
    // Positional arcs: arcs[i] corresponds to input i (TechMapper convention).
    arcs.push_back(LinearTimingArcAttr::get(ctx, delays[arcs.size()],
                                            /*sensitivity=*/0, polarity));
  }

  return MappingCostAttr::get(ctx, FloatAttr::get(Float64Type::get(ctx), area),
                              ArrayAttr::get(ctx, arcs), DictionaryAttr());
}

struct SupergateCandidate {
  hw::HWModuleOp module;
  double area;
  SmallVector<int64_t> delays;
  int64_t maxDelay;
  unsigned numGates;
  BinaryTruthTable table;
};

/// Compose a candidate function without building IR: evaluate the outer
/// cell over the inner cell (optionally inverted) and bypass inputs.
/// Building modules for every tried composition dominates generation time;
/// only dedup winners get modules below.
static BinaryTruthTable
composeSupergateTable(const BinaryTruthTable &innerTable,
                      const BinaryTruthTable &outerTable, unsigned pin,
                      ArrayRef<unsigned> innerToInput,
                      ArrayRef<unsigned> outerBypassToInput, bool invertInner,
                      unsigned totalInputs) {
  unsigned width = 1u << totalInputs;
  APInt result(width, 0);
  for (unsigned row = 0; row < width; ++row) {
    unsigned innerRow = 0;
    for (unsigned i = 0, e = innerToInput.size(); i != e; ++i)
      if ((row >> innerToInput[i]) & 1u)
        innerRow |= 1u << i;
    bool innerOut = innerTable.table[innerRow];
    if (invertInner)
      innerOut = !innerOut;
    unsigned outerRow = 0;
    unsigned bypassIdx = 0;
    for (unsigned i = 0, e = outerTable.numInputs; i != e; ++i) {
      bool bit = (i == pin) ? innerOut
                            : ((row >> outerBypassToInput[bypassIdx++]) & 1u);
      if (bit)
        outerRow |= 1u << i;
    }
    if (outerTable.table[outerRow])
      result.setBit(row);
  }
  return BinaryTruthTable(totalInputs, 1, result);
}

struct GenSupergatesPass
    : public circt::synth::impl::GenSupergatesBase<GenSupergatesPass> {
  using GenSupergatesBase::GenSupergatesBase;

  void runOnOperation() override {
    auto topModule = getOperation();
    auto *ctx = topModule.getContext();

    if (maxGates < 2)
      return;

    SmallVector<CellInfo> baseCells;
    if (failed(collectBaseCells(topModule, baseCells))) {
      signalPassFailure();
      return;
    }
    if (baseCells.empty())
      return;
    std::optional<CellInfo> inverter = findCheapestInverter(baseCells);

    DenseMap<RawTableKey, double> coveredRaw;
    for (auto &cell : baseCells) {
      auto key = makeRawKey(cell.table);
      auto it = coveredRaw.find(key);
      if (it == coveredRaw.end() || cell.area < it->second)
        coveredRaw[key] = cell.area;
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(topModule.getBody());
    Location loc = topModule.getLoc();

    DenseMap<RawTableKey, SupergateCandidate> bestByRaw;
    // Every created candidate module by its function key (for retargeting
    // references when a cheaper same-function candidate outcompetes it).
    DenseMap<Operation *, RawTableKey> moduleToKey;
    unsigned supergateOrdinal = 0;

    for (unsigned depth = 2; depth <= maxGates; ++depth) {
      SmallVector<CellInfo> innerPool = baseCells;
      for (auto &entry : bestByRaw) {
        auto &candidate = entry.second;
        if (candidate.numGates < depth)
          innerPool.push_back({candidate.module, candidate.area,
                               candidate.delays, candidate.maxDelay,
                               candidate.numGates, candidate.table});
      }
      if (innerPool.empty())
        break;

      std::sort(innerPool.begin(), innerPool.end(),
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

          for (auto &inner : innerPool) {
            if (maxCandidatesPerRoot > 0 &&
                triedForRootPin >= maxCandidatesPerRoot)
              break;

            unsigned innerInputs = inner.module.getNumInputPorts();

            auto tryCandidate = [&](ArrayRef<unsigned> innerToInput,
                                    ArrayRef<unsigned> outerBypassToInput,
                                    unsigned totalInputs, bool invertInner) {
              if (totalInputs > maxInputs)
                return;
              if (totalInputs >= maxTruthTableInputs)
                return;
              const CellInfo *innerInverter = nullptr;
              unsigned candidateNumGates = inner.numGates + outer.numGates;
              double area = inner.area + outer.area;
              int64_t viaInner = outer.delays[pin];
              if (invertInner) {
                if (!inverter)
                  return;
                innerInverter = &*inverter;
                candidateNumGates += inverter->numGates;
                if (candidateNumGates > maxGates)
                  return;
                area += inverter->area;
                viaInner += inverter->maxDelay;
              }
              if (candidateNumGates != depth)
                return;
              if (maxArea > 0.0 && area > maxArea)
                return;

              int64_t candidateMaxDelay = std::max(inner.maxDelay + viaInner,
                                                   outerMaxExcludingPin[pin]);
              if (maxDelay > 0 && candidateMaxDelay > maxDelay)
                return;

              ++triedForRootPin;
              // Compose the function first: only dedup winners get IR.
              BinaryTruthTable composed = composeSupergateTable(
                  inner.table, outer.table, pin, innerToInput,
                  outerBypassToInput, invertInner, totalInputs);

              auto key = makeRawKey(composed);
              if (coveredRaw.count(key))
                return;

              auto it = bestByRaw.find(key);
              if (it != bestByRaw.end() && area >= it->second.area)
                return;

              builder.setInsertionPointToEnd(topModule.getBody());
              SmallString<32> supergateName("__supergate_");
              supergateName += std::to_string(supergateOrdinal++);
              auto sg = createSupergateModule(
                  builder, loc, StringAttr::get(ctx, supergateName), inner,
                  outer, pin, innerToInput, outerBypassToInput, totalInputs,
                  innerInverter);


              SmallVector<int64_t> delays(totalInputs, 0);
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

              // New best (checked above): building the module already
              // consumed an ordinal; store with its composed function.
              moduleToKey[sg] = key;
              bestByRaw[key] = {sg,
                                area,
                                std::move(delays),
                                candidateMaxDelay,
                                candidateNumGates,
                                std::move(composed)};
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
                         baseTotalInputs, /*invertInner=*/false);
            tryCandidate(baseInnerToInput, baseOuterBypassToInput,
                         baseTotalInputs, /*invertInner=*/true);

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
                             baseTotalInputs - 1, /*invertInner=*/false);
                tryCandidate(dupInnerToInput, dupOuterBypassToInput,
                             baseTotalInputs - 1, /*invertInner=*/true);
                ++bypassPos;
              }
            }

            if (maxCandidatesPerRoot > 0 &&
                triedForRootPin >= maxCandidatesPerRoot)
              break;
          }
        }
      }
    }

    for (auto &entry : bestByRaw) {
      auto &candidate = entry.second;
      candidate.module->setAttr("synth.mapping_cost",
                                buildMappingCostAttr(ctx, candidate.module,
                                                     candidate.area,
                                                     candidate.delays));
      candidate.module->setAttr("synth.supergate", BoolAttr::get(ctx, true));
      SymbolTable::setSymbolVisibility(candidate.module,
                                       SymbolTable::Visibility::Public);

      LLVM_DEBUG(llvm::dbgs() << "Generated supergate: "
                              << candidate.module.getModuleName() << "\n");
    }

    // Drop rejected candidates (outcompeted winners without costs). Before
    // erasing, retarget any instances of each dropped module (referenced
    // as inner by candidates composed while it was still the winner) to
    // the surviving winner for the same function. Same key means same
    // function and input count, and all supergates share the in0..inN
    // plus Y interface, so substitution is exact.
    SmallVector<hw::HWModuleOp> deadCandidates;
    for (auto hwModule : topModule.getOps<hw::HWModuleOp>())
      if (hwModule->hasAttr("synth.supergate_candidate") &&
          !hwModule->hasAttr("synth.mapping_cost"))
        deadCandidates.push_back(hwModule);
    for (auto dead : deadCandidates) {
      auto keyIt = moduleToKey.find(dead);
      if (keyIt == moduleToKey.end())
        continue;
      auto winnerIt = bestByRaw.find(keyIt->second);
      if (winnerIt == bestByRaw.end() || winnerIt->second.module == dead)
        continue;
      hw::HWModuleOp winner = winnerIt->second.module;
      for (auto userModule : topModule.getOps<hw::HWModuleOp>()) {
        if (!userModule.getBodyBlock() || userModule == dead)
          continue;
        SmallVector<hw::InstanceOp> users;
        for (auto &op : userModule.getBodyBlock()->getOperations()) {
          auto inst = dyn_cast<hw::InstanceOp>(op);
          if (inst && inst.getModuleName() == dead.getModuleName())
            users.push_back(inst);
        }
        OpBuilder userBuilder(userModule.getContext());
        for (auto inst : users) {
          userBuilder.setInsertionPoint(inst);
          SmallVector<Value> operands(inst.getOperands().begin(),
                                      inst.getOperands().end());
          auto replacement = hw::InstanceOp::create(
              userBuilder, inst.getLoc(), winner,
              inst.getInstanceNameAttr().getValue(), operands);
          inst.getResult(0).replaceAllUsesWith(replacement.getResult(0));
          inst->erase();
        }
      }
    }
    for (auto dead : deadCandidates)
      dead.erase();


    LLVM_DEBUG(llvm::dbgs()
               << "Generated " << bestByRaw.size() << " supergates\n");
  }
};

} // namespace
