//===- AnnotateTechLib.cpp - Annotate Liberty modules with techlib info ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_ANNOTATETECHLIB
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {

/// Extract a scalar delay from NLDM table values based on the strategy.
static double extractDelay(ArrayAttr cellRiseValues, ArrayAttr cellFallValues,
                           StringRef strategy) {
  auto getMax = [](ArrayAttr values) -> double {
    double maxVal = 0.0;
    for (auto attr : values)
      if (auto f = dyn_cast<FloatAttr>(attr))
        maxVal = std::max(maxVal, f.getValue().convertToDouble());
    return maxVal;
  };

  auto getFirst = [](ArrayAttr values) -> double {
    if (values.empty())
      return 0.0;
    if (auto f = dyn_cast<FloatAttr>(values[0]))
      return f.getValue().convertToDouble();
    return 0.0;
  };

  auto getCenter = [](ArrayAttr values) -> double {
    if (values.empty())
      return 0.0;
    size_t mid = values.size() / 2;
    if (auto f = dyn_cast<FloatAttr>(values[mid]))
      return f.getValue().convertToDouble();
    return 0.0;
  };

  double riseDelay = 0.0, fallDelay = 0.0;

  if (strategy == "first") {
    riseDelay = cellRiseValues.empty() ? 0.0 : getFirst(cellRiseValues);
    fallDelay = cellFallValues.empty() ? 0.0 : getFirst(cellFallValues);
  } else if (strategy == "center") {
    riseDelay = cellRiseValues.empty() ? 0.0 : getCenter(cellRiseValues);
    fallDelay = cellFallValues.empty() ? 0.0 : getCenter(cellFallValues);
  } else {
    // worst-case (default): max of all values
    riseDelay = cellRiseValues.empty() ? 0.0 : getMax(cellRiseValues);
    fallDelay = cellFallValues.empty() ? 0.0 : getMax(cellFallValues);
  }

  return std::max(riseDelay, fallDelay);
}

struct AnnotateTechLibPass
    : public circt::synth::impl::AnnotateTechLibBase<AnnotateTechLibPass> {
  using AnnotateTechLibBase::AnnotateTechLibBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();

    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      // Skip modules that already have hw.techlib.info.
      if (hwModule->hasAttr("hw.techlib.info"))
        continue;

      // Check if this module has NLDM arcs on any output port.
      auto moduleTy = hwModule.getHWModuleType();
      bool hasNldmArcs = false;
      for (unsigned i = 0, e = moduleTy.getNumPorts(); i < e; ++i) {
        auto portAttrs = hwModule.getPort(i).attrs;
        if (!portAttrs)
          continue;
        if (auto pinDict =
                portAttrs.getAs<DictionaryAttr>("synth.liberty.pin"))
          if (pinDict.get("synth.nldm.arcs"))
            hasNldmArcs = true;
      }

      if (!hasNldmArcs)
        continue;

      // Extract area.
      auto areaAttr = hwModule->getAttrOfType<FloatAttr>("synth.liberty.area");
      double area = areaAttr ? areaAttr.getValue().convertToDouble() : 0.0;

      // Build input name list.
      SmallVector<StringRef> inputNames;
      for (auto port : moduleTy.getPorts()) {
        if (port.dir == hw::ModulePort::Direction::Input)
          inputNames.push_back(port.name.getValue());
      }

      // For each input, find the max delay across all output arcs.
      SmallVector<double> inputDelays(inputNames.size(), 0.0);

      // Scan output ports for NLDM arcs.
      for (unsigned i = 0, e = moduleTy.getNumPorts(); i < e; ++i) {
        auto port = moduleTy.getPorts()[i];
        if (port.dir != hw::ModulePort::Direction::Output)
          continue;

        auto portAttrs = hwModule.getPort(i).attrs;
        if (!portAttrs)
          continue;

        auto pinDict = portAttrs.getAs<DictionaryAttr>("synth.liberty.pin");
        if (!pinDict)
          continue;

        auto arcsAttr = pinDict.getAs<ArrayAttr>("synth.nldm.arcs");
        if (!arcsAttr)
          continue;

        for (auto arcAttr : arcsAttr) {
          auto arc = dyn_cast<synth::NLDMArcAttr>(arcAttr);
          if (!arc)
            continue;

          StringRef relatedPin = arc.getRelatedPin().getValue();

          // Find input index.
          auto it = llvm::find(inputNames, relatedPin);
          if (it == inputNames.end())
            continue;
          unsigned inputIdx = std::distance(inputNames.begin(), it);

          double delay =
              extractDelay(arc.getCellRiseValues(), arc.getCellFallValues(),
                           delayExtraction);
          inputDelays[inputIdx] = std::max(inputDelays[inputIdx], delay);
        }
      }

      // Build the delay attribute: [[d0], [d1], ...] as IntegerAttr.
      // TechMapper reads delay elements as IntegerAttr, so we scale to
      // picoseconds and round.
      auto timeUnitAttr = module->getAttrOfType<synth::NLDMTimeUnitAttr>(
          "synth.nldm.time_unit");
      double timeScale =
          timeUnitAttr
              ? timeUnitAttr.getPicoseconds().getValue().convertToDouble()
              : 1.0;

      SmallVector<Attribute> delayPerInput;
      for (double d : inputDelays) {
        int64_t delayPs = static_cast<int64_t>(std::round(d * timeScale));
        SmallVector<Attribute> inner;
        inner.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), delayPs));
        delayPerInput.push_back(ArrayAttr::get(ctx, inner));
      }

      // Attach hw.techlib.info.
      SmallVector<NamedAttribute> techInfoAttrs;
      techInfoAttrs.push_back(
          NamedAttribute(StringAttr::get(ctx, "area"),
                         FloatAttr::get(Float64Type::get(ctx), area)));
      techInfoAttrs.push_back(NamedAttribute(StringAttr::get(ctx, "delay"),
                                             ArrayAttr::get(ctx, delayPerInput)));

      hwModule->setAttr("hw.techlib.info",
                        DictionaryAttr::get(ctx, techInfoAttrs));
    }
  }
};

} // namespace
