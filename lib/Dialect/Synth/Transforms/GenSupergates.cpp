//===- GenSupergates.cpp - Generate supergate library
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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
  SmallVector<int64_t> delays;
  unsigned numGates;
};

static void collectBaseCells(ModuleOp topModule,
                             SmallVectorImpl<CellInfo> &cells) {
  for (auto hwModule : topModule.getOps<hw::HWModuleOp>()) {
    if (hwModule->hasAttr("synth.supergate"))
      continue;

    auto techInfo = hwModule->getAttrOfType<DictionaryAttr>("hw.techlib.info");
    if (!techInfo)
      continue;

    auto areaAttr = techInfo.getAs<FloatAttr>("area");
    auto delayAttr = techInfo.getAs<ArrayAttr>("delay");
    if (!areaAttr || !delayAttr)
      continue;

    CellInfo info;
    info.module = hwModule;
    info.area = areaAttr.getValue().convertToDouble();
    for (auto perInputAttr : delayAttr) {
      auto perInput = dyn_cast<ArrayAttr>(perInputAttr);
      if (!perInput || perInput.empty())
        continue;
      auto d = dyn_cast<IntegerAttr>(perInput[0]);
      if (!d)
        continue;
      info.delays.push_back(d.getValue().getSExtValue());
    }

    if (info.delays.size() != hwModule.getNumInputPorts())
      continue;
    info.numGates = 1;
    cells.push_back(std::move(info));
  }
}

static FailureOr<NPNClass> computeNPN(hw::HWModuleOp module,
                                      igraph::InstanceGraph *instanceGraph) {
  if (module.getNumOutputPorts() != 1)
    return failure();
  if (!module.getOutputTypes()[0].isInteger(1))
    return failure();
  if (module.getNumInputPorts() > synth::maxTruthTableInputs)
    return failure();

  for (auto inputType : module.getInputTypes())
    if (!inputType.isInteger(1))
      return failure();

  auto *body = module.getBodyBlock();
  auto *terminator = body->getTerminator();
  auto outputOp = dyn_cast_or_null<hw::OutputOp>(terminator);
  if (!outputOp)
    return failure();

  auto tt = synth::getTruthTable(outputOp.getOutputs(), body, instanceGraph);
  if (failed(tt))
    return failure();
  return NPNClass::computeNPNCanonicalForm(*tt);
}

static hw::HWModuleOp createSupergateModule(OpBuilder &builder, Location loc,
                                            StringAttr name, CellInfo &inner,
                                            CellInfo &outer, unsigned pin) {
  auto savedInsertion = builder.saveInsertionPoint();
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;

  unsigned portIdx = 0;
  for (unsigned i = 0, e = inner.module.getNumInputPorts(); i != e; ++i) {
    auto inName = inner.module.getInputName(i);
    ports.push_back(
        {{StringAttr::get(ctx, inName), i1, hw::ModulePort::Direction::Input},
         portIdx++});
  }
  for (unsigned i = 0, e = outer.module.getNumInputPorts(); i != e; ++i) {
    if (i == pin)
      continue;
    auto inName = outer.module.getInputName(i);
    SmallString<32> unique(inName);
    unique += "_outer";
    ports.push_back(
        {{StringAttr::get(ctx, unique), i1, hw::ModulePort::Direction::Input},
         portIdx++});
  }

  ports.push_back(
      {{StringAttr::get(ctx, "Y"), i1, hw::ModulePort::Direction::Output},
       portIdx});

  auto sg = hw::HWModuleOp::create(builder, loc, name, ports);
  sg.setPrivate();

  auto *body = sg.getBodyBlock();
  if (body->mightHaveTerminator())
    body->getTerminator()->erase();

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> innerOperands;
  for (unsigned i = 0, e = inner.module.getNumInputPorts(); i != e; ++i)
    innerOperands.push_back(body->getArgument(i));

  auto innerInst = hw::InstanceOp::create(builder, loc, inner.module, "inner",
                                          innerOperands);

  SmallVector<Value> outerOperands;
  unsigned nextArg = inner.module.getNumInputPorts();
  for (unsigned i = 0, e = outer.module.getNumInputPorts(); i != e; ++i) {
    if (i == pin) {
      outerOperands.push_back(innerInst.getResult(0));
      continue;
    }
    outerOperands.push_back(body->getArgument(nextArg++));
  }

  auto outerInst = hw::InstanceOp::create(builder, loc, outer.module, "outer",
                                          outerOperands);
  hw::OutputOp::create(builder, loc, ValueRange{outerInst.getResult(0)});
  builder.restoreInsertionPoint(savedInsertion);
  return sg;
}

static ArrayAttr buildDelayAttr(MLIRContext *ctx, ArrayRef<int64_t> delays) {
  SmallVector<Attribute> delayPerInput;
  for (int64_t d : delays) {
    auto delayValue = IntegerAttr::get(IntegerType::get(ctx, 64), d);
    delayPerInput.push_back(ArrayAttr::get(ctx, delayValue));
  }
  return ArrayAttr::get(ctx, delayPerInput);
}

struct SupergateCandidate {
  hw::HWModuleOp module;
  double area;
  SmallVector<int64_t> delays;
  unsigned numGates;
};

struct GenSupergatesPass
    : public circt::synth::impl::GenSupergatesBase<GenSupergatesPass> {
  using GenSupergatesBase::GenSupergatesBase;

  void runOnOperation() override {
    auto topModule = getOperation();
    auto *ctx = topModule.getContext();
    auto &instanceGraph = getAnalysis<igraph::InstanceGraph>();

    if (maxGates < 2)
      return;

    SmallVector<CellInfo> baseCells;
    collectBaseCells(topModule, baseCells);
    if (baseCells.empty())
      return;

    DenseMap<llvm::APInt, double> coveredNPN;
    for (auto &cell : baseCells) {
      auto npn = computeNPN(cell.module, &instanceGraph);
      if (failed(npn))
        continue;
      auto key = npn->truthTable.table;
      auto it = coveredNPN.find(key);
      if (it == coveredNPN.end() || cell.area < it->second)
        coveredNPN[key] = cell.area;
    }

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(topModule.getBody());
    Location loc = topModule.getLoc();

    DenseMap<llvm::APInt, SupergateCandidate> bestByNPN;
    unsigned supergateOrdinal = 0;

    SmallVector<CellInfo> frontier = baseCells;
    for (unsigned depth = 2; depth <= maxGates && !frontier.empty(); ++depth) {
      for (auto &inner : frontier) {
        for (auto &outer : baseCells) {
          unsigned innerInputs = inner.module.getNumInputPorts();
          unsigned outerInputs = outer.module.getNumInputPorts();

          double area = inner.area + outer.area;
          if (maxArea > 0.0 && area > maxArea)
            continue;

          for (unsigned pin = 0; pin < outerInputs; ++pin) {
            unsigned totalInputs = innerInputs + outerInputs - 1;
            if (totalInputs > maxInputs)
              continue;

            builder.setInsertionPointToEnd(topModule.getBody());
            SmallString<32> supergateName("__supergate_");
            supergateName += std::to_string(supergateOrdinal++);
            auto sg = createSupergateModule(builder, loc,
                                            StringAttr::get(ctx, supergateName),
                                            inner, outer, pin);
            instanceGraph.addModule(sg);

            auto npn = computeNPN(sg, &instanceGraph);
            if (failed(npn))
              continue;

            auto key = npn->truthTable.table;
            if (coveredNPN.count(key))
              continue;

            SmallVector<int64_t> delays;
            delays.reserve(totalInputs);
            int64_t viaInner = outer.delays[pin];
            for (unsigned i = 0; i < innerInputs; ++i)
              delays.push_back(inner.delays[i] + viaInner);
            for (unsigned i = 0; i < outerInputs; ++i) {
              if (i == pin)
                continue;
              delays.push_back(outer.delays[i]);
            }

            auto it = bestByNPN.find(key);
            if (it == bestByNPN.end() || area < it->second.area)
              bestByNPN[key] = {sg, area, std::move(delays), depth};
          }
        }
      }

      SmallVector<CellInfo> nextFrontier;
      for (auto &entry : bestByNPN) {
        auto &candidate = entry.second;
        if (candidate.numGates != depth)
          continue;
        nextFrontier.push_back({candidate.module, candidate.area,
                                candidate.delays, candidate.numGates});
      }
      frontier = std::move(nextFrontier);
    }

    for (auto &entry : bestByNPN) {
      auto &candidate = entry.second;
      SmallVector<NamedAttribute> techInfo;
      techInfo.push_back(
          {StringAttr::get(ctx, "area"),
           FloatAttr::get(Float64Type::get(ctx), candidate.area)});
      techInfo.push_back({StringAttr::get(ctx, "delay"),
                          buildDelayAttr(ctx, candidate.delays)});
      candidate.module->setAttr("hw.techlib.info",
                                DictionaryAttr::get(ctx, techInfo));
      candidate.module->setAttr("synth.supergate", BoolAttr::get(ctx, true));

      LLVM_DEBUG(llvm::dbgs() << "Generated supergate: "
                              << candidate.module.getModuleName() << "\n");
    }

    llvm::SmallPtrSet<Operation *, 16> liveSupergates;
    for (auto &entry : bestByNPN)
      liveSupergates.insert(entry.second.module.getOperation());

    llvm::SmallPtrSet<Operation *, 16> erasedModules;

    // Drop generated modules that were not selected as best representatives.
    // We intentionally do not update InstanceGraph at the end of the pass since
    // this analysis result is not preserved.
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
