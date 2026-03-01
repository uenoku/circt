//===- DelayModel.cpp - Pluggable Delay Model Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/Timing/Liberty.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <optional>

using namespace circt;
using namespace circt::synth::timing;

static int64_t getAttrDelay(Operation *op, StringRef attrName) {
  if (!op)
    return -1;
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return -1;
}

static int64_t getPerArcDelay(Operation *op, int32_t inputIndex,
                              int32_t outputIndex) {
  if (!op)
    return -1;

  auto dict = op->getAttrOfType<DictionaryAttr>("synth.liberty.arc_delay_ps");
  if (!dict)
    return -1;

  if (inputIndex < 0 || outputIndex < 0)
    return -1;

  llvm::SmallString<32> key;
  key += "i";
  key += llvm::utostr(static_cast<unsigned>(inputIndex));
  key += "_o";
  key += llvm::utostr(static_cast<unsigned>(outputIndex));

  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(key)))
    return attr.getInt();
  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get("default")))
    return attr.getInt();
  return -1;
}

static int64_t getPerArcDelayByPin(Operation *op, StringRef inputPin,
                                   StringRef outputPin) {
  if (!op)
    return -1;

  auto dict = op->getAttrOfType<DictionaryAttr>("synth.liberty.arc_delay_ps");
  if (!dict)
    return -1;

  llvm::SmallString<64> key;
  key += inputPin;
  key += "_to_";
  key += outputPin;

  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(key)))
    return attr.getInt();
  if (auto attr = dyn_cast_or_null<IntegerAttr>(dict.get("default")))
    return attr.getInt();
  return -1;
}

static std::optional<StringRef> getMappedCellName(Operation *op) {
  if (!op)
    return std::nullopt;

  if (auto mapped = op->getAttrOfType<StringAttr>("synth.liberty.cell"))
    return mapped.getValue();

  if (auto inst = dyn_cast<hw::InstanceOp>(op))
    return inst.getReferencedModuleName();

  return std::nullopt;
}

static double getTimeScalePs(Operation *op) {
  if (!op)
    return 1.0;

  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return 1.0;

  if (auto unit = module->getAttrOfType<synth::NLDMTimeUnitAttr>(
          "synth.nldm.time_unit"))
    return unit.getPicoseconds().getValueAsDouble();
  return 1.0;
}

static std::optional<double> getFirstNumericAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValueAsDouble();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return static_cast<double>(intAttr.getInt());
  if (auto arr = dyn_cast<ArrayAttr>(attr)) {
    if (arr.empty())
      return std::nullopt;
    return getFirstNumericAttr(arr[0]);
  }
  return std::nullopt;
}

static std::optional<int64_t>
getDelayFromTimingArc(synth::NLDMArcAttr timingArc, double timeScalePs) {
  if (auto rise = getFirstNumericAttr(timingArc.getCellRiseValues()))
    return static_cast<int64_t>(std::llround(*rise * timeScalePs));
  if (auto fall = getFirstNumericAttr(timingArc.getCellFallValues()))
    return static_cast<int64_t>(std::llround(*fall * timeScalePs));
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// UnitDelayModel
//===----------------------------------------------------------------------===//

DelayResult UnitDelayModel::computeDelay(const DelayContext &ctx) const {
  auto *op = ctx.op;
  // Zero-cost wiring operations
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op))
    return {0, 0.0};
  // All other ops get unit delay
  return {1, 0.0};
}

//===----------------------------------------------------------------------===//
// AIGLevelDelayModel
//===----------------------------------------------------------------------===//

DelayResult AIGLevelDelayModel::computeDelay(const DelayContext &ctx) const {
  auto *op = ctx.op;

  // AIG operations
  if (auto andOp = dyn_cast<aig::AndInverterOp>(op))
    return {llvm::Log2_64_Ceil(andOp.getNumOperands()), 0.0};

  // Comb operations
  if (isa<comb::MuxOp>(op))
    return {1, 0.0};
  if (auto andOp = dyn_cast<comb::AndOp>(op))
    return {llvm::Log2_64_Ceil(andOp.getNumOperands()), 0.0};
  if (auto orOp = dyn_cast<comb::OrOp>(op))
    return {llvm::Log2_64_Ceil(orOp.getNumOperands()), 0.0};
  if (auto xorOp = dyn_cast<comb::XorOp>(op))
    return {llvm::Log2_64_Ceil(xorOp.getNumOperands()), 0.0};

  // Zero-cost operations (bit manipulation)
  if (isa<comb::ConcatOp, comb::ExtractOp, comb::ReplicateOp>(op))
    return {0, 0.0};

  // Default cost
  return {1, 0.0};
}

//===----------------------------------------------------------------------===//
// NLDMDelayModel
//===----------------------------------------------------------------------===//

NLDMDelayModel::NLDMDelayModel() = default;

NLDMDelayModel::NLDMDelayModel(std::unique_ptr<LibertyLibrary> liberty)
    : liberty(std::move(liberty)) {}

NLDMDelayModel::~NLDMDelayModel() = default;

DelayResult NLDMDelayModel::computeDelay(const DelayContext &ctx) const {
  if (int64_t delay = getPerArcDelay(ctx.op, ctx.inputIndex, ctx.outputIndex);
      delay >= 0)
    return {delay, 0.0};

  if (liberty && ctx.inputIndex >= 0 && ctx.outputIndex >= 0) {
    if (auto cellName = getMappedCellName(ctx.op)) {
      auto inputPin = liberty->getInputPinName(*cellName, ctx.inputIndex);
      auto outputPin = liberty->getOutputPinName(*cellName, ctx.outputIndex);
      if (inputPin && outputPin) {
        if (auto timingArc = liberty->getTypedTimingArc(
                *cellName, ctx.inputIndex, ctx.outputIndex)) {
          if (auto delay =
                  getDelayFromTimingArc(*timingArc, getTimeScalePs(ctx.op)))
            return {*delay, 0.0};
        }

        if (int64_t delay = getPerArcDelayByPin(ctx.op, *inputPin, *outputPin);
            delay >= 0)
          return {delay, 0.0};
      }
    }
  }

  if (int64_t delay = getAttrDelay(ctx.op, "synth.liberty.delay_ps");
      delay >= 0)
    return {delay, 0.0};

  return fallback.computeDelay(ctx);
}

double NLDMDelayModel::getInputCapacitance(const DelayContext &ctx) const {
  if (!liberty || ctx.inputIndex < 0)
    return 0.0;

  auto cellName = getMappedCellName(ctx.op);
  if (!cellName)
    return 0.0;

  auto cap = liberty->getInputPinCapacitance(
      *cellName, static_cast<unsigned>(ctx.inputIndex));
  return cap.value_or(0.0);
}

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<DelayModel> circt::synth::timing::createDefaultDelayModel() {
  return std::make_unique<AIGLevelDelayModel>();
}

std::unique_ptr<DelayModel> circt::synth::timing::createNLDMDelayModel() {
  return std::make_unique<NLDMDelayModel>();
}

std::unique_ptr<DelayModel>
circt::synth::timing::createNLDMDelayModel(ModuleOp module) {
  if (!module)
    return std::make_unique<NLDMDelayModel>();

  auto libertyOr = LibertyLibrary::fromModule(module);
  if (failed(libertyOr))
    return std::make_unique<NLDMDelayModel>();

  return std::make_unique<NLDMDelayModel>(
      std::make_unique<LibertyLibrary>(std::move(*libertyOr)));
}
