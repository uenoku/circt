//===- Liberty.cpp - Liberty Data Bridge for Timing Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/Liberty.h"
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
