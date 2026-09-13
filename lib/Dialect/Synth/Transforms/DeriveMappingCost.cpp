//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DeriveMappingCost pass, which derives
// `synth.mapping_cost` attributes for imported Liberty combinational cells
// from their `synth.timing.nldm` arcs and cell `area`, so that
// `synth-tech-mapper` can use them as technology patterns.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cmath>

namespace circt {
namespace synth {
#define GEN_PASS_DEF_DERIVEMAPPINGCOST
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace mlir;

#define DEBUG_TYPE "synth-derive-mapping-cost"

namespace {

// Time-unit scale: NLDM values are in nanoseconds, the integer cost model
// uses picoseconds.
constexpr double kNsToPs = 1000.0;

struct DeriveMappingCostPass
    : public circt::synth::impl::DeriveMappingCostBase<
          DeriveMappingCostPass> {
  using DeriveMappingCostBase<DeriveMappingCostPass>::DeriveMappingCostBase;

  void runOnOperation() override {
    auto top = getOperation();
    llvm::DenseSet<StringRef> dontUseSet(dontUse.begin(), dontUse.end());
    OpBuilder builder(top.getContext());

    for (auto module : top.getOps<hw::HWModuleOp>()) {
      auto skip = [&] { ++numCostsSkipped; };
      // Stateful bodies are matched by sequential mapping instead.
      bool hasState = module.getBodyBlock()
                          ->walk([](Operation *op) {
                            if (isa<FFOp, LatchOp, ClockGateOp>(op))
                              return WalkResult::interrupt();
                            return WalkResult::advance();
                          })
                          .wasInterrupted();
      if (hasState) {
        skip();
        continue;
      }
      if (dontUseSet.count(module.getModuleName())) {
        module->setAttr("synth.dont_use", builder.getUnitAttr());
        skip();
        continue;
      }
      auto nldm = module->getAttrOfType<ArrayAttr>("synth.timing.nldm");
      if (!nldm || nldm.empty()) {
        skip();
        continue;
      }
      // Patterns need a single all-i1 output and all-i1 inputs.
      auto inputTypes = module.getInputTypes();
      auto outputTypes = module.getOutputTypes();
      if (outputTypes.size() != 1 || !outputTypes[0].isInteger(1)) {
        skip();
        continue;
      }
      bool allSingleBit = llvm::all_of(inputTypes, [](Type t) {
        return t.isInteger(1);
      });
      if (!allSingleBit) {
        skip();
        continue;
      }
      // Area from the preserved Liberty cell attributes.
      FloatAttr area;
      if (auto cellDict =
              module->getAttrOfType<DictionaryAttr>("synth.liberty.cell"))
        area = dyn_cast_or_null<FloatAttr>(cellDict.get("area"));
      if (!area) {
        skip();
        continue;
      }

      // One linear arc per input port, in port order.
      SmallVector<Attribute> arcs;
      SmallVector<NamedAttribute> inputCapEntries;
      bool complete = true;
      for (const auto &port : module.getPortList()) {
        if (!port.isInput())
          continue;
        double sum = 0.0;
        size_t count = 0;
        bool negative = true;
        bool hasArc = false;
        for (auto arcAttr : nldm) {
          auto arc = cast<NLDMTimingArcAttr>(arcAttr);
          if (arc.getRelatedPin() != port.name)
            continue;
          hasArc = true;
          for (auto table : {arc.getCellRise(), arc.getCellFall()}) {
            for (double v : table.getValues().asArrayRef()) {
              sum += v;
              ++count;
            }
          }
          if (arc.getTimingSense().getValue() != "negative_unate")
            negative = false;
        }
        if (!hasArc || count == 0) {
          complete = false;
          break;
        }
        int64_t intrinsic =
            std::max<int64_t>(1, llround(kNsToPs * sum / count));
        auto polarity = PolarityAttr::get(
            builder.getContext(), negative ? PolarityKind::Negative
                                           : PolarityKind::Positive);
        arcs.push_back(LinearTimingArcAttr::get(builder.getContext(), intrinsic,
                                                /*sensitivity=*/0, polarity));
        if (port.attrs) {
          if (auto pinDict = dyn_cast_or_null<DictionaryAttr>(
                  port.attrs.get("synth.liberty.pin"))) {
            if (auto cap =
                    dyn_cast_or_null<FloatAttr>(pinDict.get("capacitance")))
              inputCapEntries.push_back(builder.getNamedAttr(port.name, cap));
          }
        }
      }
      if (!complete) {
        skip();
        continue;
      }
      DictionaryAttr inputCaps =
          inputCapEntries.empty()
              ? DictionaryAttr()
              : builder.getDictionaryAttr(inputCapEntries);
      module->setAttr("synth.mapping_cost",
                      MappingCostAttr::get(builder.getContext(), area,
                                           builder.getArrayAttr(arcs),
                                           inputCaps));
      ++numCostsDerived;
    }
  }
};

} // namespace
