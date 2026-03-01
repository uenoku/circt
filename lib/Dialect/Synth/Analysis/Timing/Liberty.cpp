//===- Liberty.cpp - Liberty Data Bridge for Timing Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/Liberty.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace circt::synth::timing;

static std::optional<double> parseNumericAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValueAsDouble();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return static_cast<double>(intAttr.getInt());
  if (auto strAttr = dyn_cast<StringAttr>(attr)) {
    double value = 0.0;
    if (!strAttr.getValue().getAsDouble(value))
      return value;
  }
  return std::nullopt;
}

FailureOr<LibertyLibrary> LibertyLibrary::fromModule(ModuleOp module) {
  LibertyLibrary view;

  for (auto hwMod : module.getOps<hw::HWModuleOp>()) {
    auto portList = hwMod.getPortList();

    bool hasLibertyPinAttrs = false;
    Cell cell;
    cell.module = hwMod;

    for (auto [idx, port] : llvm::enumerate(portList)) {
      auto portAttrs =
          dyn_cast_or_null<DictionaryAttr>(hwMod.getPortAttrs(idx));
      if (!portAttrs)
        continue;

      auto pinAttrs =
          dyn_cast_or_null<DictionaryAttr>(portAttrs.get("synth.liberty.pin"));
      if (!pinAttrs)
        continue;

      hasLibertyPinAttrs = true;

      Pin pin;
      pin.isInput = port.isInput();
      pin.capacitance = parseNumericAttr(pinAttrs.get("capacitance"));
      pin.attrs = pinAttrs;
      auto pinName = port.name.getValue().str();
      cell.pins[pinName] = std::move(pin);
      if (port.isInput())
        cell.inputPinsByIndex.push_back(pinName);
      else if (port.isOutput())
        cell.outputPinsByIndex.push_back(pinName);
    }

    if (!hasLibertyPinAttrs)
      continue;

    view.cells[hwMod.getModuleName()] = std::move(cell);
  }

  return view;
}

const LibertyLibrary::Cell *
LibertyLibrary::lookupCell(StringRef cellName) const {
  auto it = cells.find(cellName);
  if (it == cells.end())
    return nullptr;
  return &it->second;
}

const LibertyLibrary::Pin *LibertyLibrary::lookupPin(StringRef cellName,
                                                     StringRef pinName) const {
  auto *cell = lookupCell(cellName);
  if (!cell)
    return nullptr;

  auto it = cell->pins.find(pinName);
  if (it == cell->pins.end())
    return nullptr;
  return &it->second;
}

std::optional<double>
LibertyLibrary::getInputPinCapacitance(StringRef cellName,
                                       StringRef pinName) const {
  auto *pin = lookupPin(cellName, pinName);
  if (!pin || !pin->isInput || !pin->capacitance)
    return std::nullopt;
  return pin->capacitance;
}

std::optional<StringRef>
LibertyLibrary::getInputPinName(StringRef cellName,
                                unsigned operandIndex) const {
  auto *cell = lookupCell(cellName);
  if (!cell || operandIndex >= cell->inputPinsByIndex.size())
    return std::nullopt;
  return StringRef(cell->inputPinsByIndex[operandIndex]);
}

std::optional<StringRef>
LibertyLibrary::getOutputPinName(StringRef cellName,
                                 unsigned resultIndex) const {
  auto *cell = lookupCell(cellName);
  if (!cell || resultIndex >= cell->outputPinsByIndex.size())
    return std::nullopt;
  return StringRef(cell->outputPinsByIndex[resultIndex]);
}

std::optional<double>
LibertyLibrary::getInputPinCapacitance(StringRef cellName,
                                       unsigned operandIndex) const {
  auto pinName = getInputPinName(cellName, operandIndex);
  if (!pinName)
    return std::nullopt;
  return getInputPinCapacitance(cellName, *pinName);
}

std::optional<DictionaryAttr>
LibertyLibrary::getTimingArc(StringRef cellName, StringRef inputPinName,
                             StringRef outputPinName) const {
  auto typedArc = getTypedTimingArc(cellName, inputPinName, outputPinName);
  if (!typedArc)
    return std::nullopt;

  auto *outputPin = lookupPin(cellName, outputPinName);
  assert(outputPin && "typed timing arc implies valid output pin");
  MLIRContext *ctx = outputPin->attrs.getContext();

  SmallVector<NamedAttribute> attrs;
  attrs.push_back(NamedAttribute(StringAttr::get(ctx, "related_pin"),
                                 typedArc->getRelatedPin()));
  attrs.push_back(
      NamedAttribute(StringAttr::get(ctx, "to_pin"), typedArc->getToPin()));
  if (!typedArc->getTimingSense().getValue().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "timing_sense"),
                                   typedArc->getTimingSense()));
  if (!typedArc->getCellRiseIndex1().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_rise_index_1"),
                                   typedArc->getCellRiseIndex1()));
  if (!typedArc->getCellRiseIndex2().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_rise_index_2"),
                                   typedArc->getCellRiseIndex2()));
  if (!typedArc->getCellRiseValues().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_rise_values"),
                                   typedArc->getCellRiseValues()));
  if (!typedArc->getCellFallIndex1().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_fall_index_1"),
                                   typedArc->getCellFallIndex1()));
  if (!typedArc->getCellFallIndex2().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_fall_index_2"),
                                   typedArc->getCellFallIndex2()));
  if (!typedArc->getCellFallValues().empty())
    attrs.push_back(NamedAttribute(StringAttr::get(ctx, "cell_fall_values"),
                                   typedArc->getCellFallValues()));
  if (!typedArc->getRiseTransitionIndex1().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "rise_transition_index_1"),
                       typedArc->getRiseTransitionIndex1()));
  if (!typedArc->getRiseTransitionIndex2().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "rise_transition_index_2"),
                       typedArc->getRiseTransitionIndex2()));
  if (!typedArc->getRiseTransitionValues().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "rise_transition_values"),
                       typedArc->getRiseTransitionValues()));
  if (!typedArc->getFallTransitionIndex1().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "fall_transition_index_1"),
                       typedArc->getFallTransitionIndex1()));
  if (!typedArc->getFallTransitionIndex2().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "fall_transition_index_2"),
                       typedArc->getFallTransitionIndex2()));
  if (!typedArc->getFallTransitionValues().empty())
    attrs.push_back(
        NamedAttribute(StringAttr::get(ctx, "fall_transition_values"),
                       typedArc->getFallTransitionValues()));
  return DictionaryAttr::get(ctx, attrs);
}

std::optional<DictionaryAttr>
LibertyLibrary::getTimingArc(StringRef cellName, unsigned operandIndex,
                             unsigned resultIndex) const {
  auto inputPin = getInputPinName(cellName, operandIndex);
  auto outputPin = getOutputPinName(cellName, resultIndex);
  if (!inputPin || !outputPin)
    return std::nullopt;
  return getTimingArc(cellName, *inputPin, *outputPin);
}

std::optional<synth::NLDMArcAttr>
LibertyLibrary::getTypedTimingArc(StringRef cellName, StringRef inputPinName,
                                  StringRef outputPinName) const {
  auto *outputPin = lookupPin(cellName, outputPinName);
  if (!outputPin || outputPin->isInput)
    return std::nullopt;

  auto nldmArcs =
      dyn_cast_or_null<ArrayAttr>(outputPin->attrs.get("synth.nldm.arcs"));
  if (!nldmArcs)
    return std::nullopt;

  for (auto attr : nldmArcs) {
    auto typedArc = dyn_cast<synth::NLDMArcAttr>(attr);
    if (typedArc && typedArc.getRelatedPin() == inputPinName)
      return typedArc;
  }
  return std::nullopt;
}

std::optional<synth::NLDMArcAttr>
LibertyLibrary::getTypedTimingArc(StringRef cellName, unsigned operandIndex,
                                  unsigned resultIndex) const {
  auto inputPin = getInputPinName(cellName, operandIndex);
  auto outputPin = getOutputPinName(cellName, resultIndex);
  if (!inputPin || !outputPin)
    return std::nullopt;
  return getTypedTimingArc(cellName, *inputPin, *outputPin);
}

std::optional<synth::CCSPilotArcAttr>
LibertyLibrary::getTypedCCSPilotArc(StringRef cellName, StringRef inputPinName,
                                    StringRef outputPinName) const {
  auto *outputPin = lookupPin(cellName, outputPinName);
  if (!outputPin || outputPin->isInput)
    return std::nullopt;

  auto ccsArcs =
      dyn_cast_or_null<ArrayAttr>(outputPin->attrs.get("synth.ccs.pilot.arcs"));
  if (!ccsArcs)
    return std::nullopt;

  for (auto attr : ccsArcs) {
    auto typedArc = dyn_cast<synth::CCSPilotArcAttr>(attr);
    if (typedArc && typedArc.getRelatedPin() == inputPinName)
      return typedArc;
  }
  return std::nullopt;
}

std::optional<synth::CCSPilotArcAttr>
LibertyLibrary::getTypedCCSPilotArc(StringRef cellName, unsigned operandIndex,
                                    unsigned resultIndex) const {
  auto inputPin = getInputPinName(cellName, operandIndex);
  auto outputPin = getOutputPinName(cellName, resultIndex);
  if (!inputPin || !outputPin)
    return std::nullopt;
  return getTypedCCSPilotArc(cellName, *inputPin, *outputPin);
}

std::optional<synth::CCSPilotReceiverAttr>
LibertyLibrary::getTypedCCSPilotReceiver(StringRef cellName,
                                         StringRef inputPinName,
                                         StringRef outputPinName) const {
  auto *outputPin = lookupPin(cellName, outputPinName);
  if (!outputPin || outputPin->isInput)
    return std::nullopt;

  auto receivers = dyn_cast_or_null<ArrayAttr>(
      outputPin->attrs.get("synth.ccs.pilot.receivers"));
  if (!receivers)
    return std::nullopt;

  for (auto attr : receivers) {
    auto typed = dyn_cast<synth::CCSPilotReceiverAttr>(attr);
    if (typed && typed.getRelatedPin() == inputPinName)
      return typed;
  }
  return std::nullopt;
}

std::optional<synth::CCSPilotReceiverAttr>
LibertyLibrary::getTypedCCSPilotReceiver(StringRef cellName,
                                         unsigned operandIndex,
                                         unsigned resultIndex) const {
  auto inputPin = getInputPinName(cellName, operandIndex);
  auto outputPin = getOutputPinName(cellName, resultIndex);
  if (!inputPin || !outputPin)
    return std::nullopt;
  return getTypedCCSPilotReceiver(cellName, *inputPin, *outputPin);
}
