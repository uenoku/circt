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
#include <algorithm>
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

static bool decodeNumericArray(ArrayAttr array, SmallVectorImpl<double> &out) {
  out.clear();
  out.reserve(array.size());
  for (auto attr : array) {
    auto value = getFirstNumericAttr(attr);
    if (!value)
      return false;
    out.push_back(*value);
  }
  return true;
}

static double interpolate1D(double x, ArrayRef<double> xs,
                            ArrayRef<double> ys) {
  if (xs.empty() || ys.empty())
    return 0.0;
  if (xs.size() == 1 || ys.size() == 1)
    return ys.front();

  size_t n = std::min(xs.size(), ys.size());
  xs = xs.take_front(n);
  ys = ys.take_front(n);

  if (x <= xs.front())
    return ys.front();
  if (x >= xs.back())
    return ys.back();

  for (size_t i = 1; i < n; ++i) {
    if (x > xs[i])
      continue;
    double x0 = xs[i - 1];
    double x1 = xs[i];
    double y0 = ys[i - 1];
    double y1 = ys[i];
    if (x1 == x0)
      return y0;
    double t = (x - x0) / (x1 - x0);
    return y0 + (y1 - y0) * t;
  }
  return ys.back();
}

static double bilinearInterpolate(double x, double y, ArrayRef<double> xs,
                                  ArrayRef<double> ys,
                                  ArrayRef<double> values) {
  if (xs.empty() || ys.empty() || values.empty())
    return 0.0;

  auto clampToRange = [](double v, ArrayRef<double> axis) {
    if (axis.empty())
      return v;
    if (v < axis.front())
      return axis.front();
    if (v > axis.back())
      return axis.back();
    return v;
  };

  x = clampToRange(x, xs);
  y = clampToRange(y, ys);

  auto findSegment = [](double v, ArrayRef<double> axis) {
    size_t hi = 1;
    while (hi < axis.size() && v > axis[hi])
      ++hi;
    if (hi >= axis.size())
      hi = axis.size() - 1;
    size_t lo = hi - 1;
    return std::make_pair(lo, hi);
  };

  if (xs.size() == 1 && ys.size() == 1)
    return values.front();
  if (xs.size() == 1) {
    SmallVector<double> row;
    row.reserve(ys.size());
    for (size_t c = 0; c < ys.size(); ++c)
      row.push_back(values[c]);
    return interpolate1D(y, ys, row);
  }
  if (ys.size() == 1) {
    SmallVector<double> col;
    col.reserve(xs.size());
    for (size_t r = 0; r < xs.size(); ++r)
      col.push_back(values[r * ys.size()]);
    return interpolate1D(x, xs, col);
  }

  auto [x0i, x1i] = findSegment(x, xs);
  auto [y0i, y1i] = findSegment(y, ys);

  auto at = [&](size_t xi, size_t yi) { return values[xi * ys.size() + yi]; };

  double x0 = xs[x0i], x1 = xs[x1i];
  double y0 = ys[y0i], y1 = ys[y1i];
  double q00 = at(x0i, y0i);
  double q01 = at(x0i, y1i);
  double q10 = at(x1i, y0i);
  double q11 = at(x1i, y1i);

  double tx = (x1 == x0) ? 0.0 : (x - x0) / (x1 - x0);
  double ty = (y1 == y0) ? 0.0 : (y - y0) / (y1 - y0);

  double a = q00 + (q10 - q00) * tx;
  double b = q01 + (q11 - q01) * tx;
  return a + (b - a) * ty;
}

static std::optional<double> interpolateTable(double x, double y,
                                              ArrayAttr idx1Attr,
                                              ArrayAttr idx2Attr,
                                              ArrayAttr valuesAttr) {
  SmallVector<double> idx1, idx2, values;
  if (!decodeNumericArray(valuesAttr, values) || values.empty())
    return std::nullopt;

  if (!idx1Attr.empty() && !decodeNumericArray(idx1Attr, idx1))
    return std::nullopt;
  if (!idx2Attr.empty() && !decodeNumericArray(idx2Attr, idx2))
    return std::nullopt;

  if (!idx1.empty() && !idx2.empty()) {
    if (values.size() != idx1.size() * idx2.size())
      return std::nullopt;
    return bilinearInterpolate(x, y, idx1, idx2, values);
  }

  if (!idx1.empty() && values.size() == idx1.size())
    return interpolate1D(x, idx1, values);
  if (!idx2.empty() && values.size() == idx2.size())
    return interpolate1D(y, idx2, values);

  return values.front();
}

static std::optional<int64_t>
getDelayFromTimingArc(synth::NLDMArcAttr timingArc, double inputSlew,
                      double outputLoad, double timeScalePs) {
  if (auto rise = interpolateTable(
          inputSlew, outputLoad, timingArc.getCellRiseIndex1(),
          timingArc.getCellRiseIndex2(), timingArc.getCellRiseValues()))
    return static_cast<int64_t>(std::llround(*rise * timeScalePs));
  if (auto fall = interpolateTable(
          inputSlew, outputLoad, timingArc.getCellFallIndex1(),
          timingArc.getCellFallIndex2(), timingArc.getCellFallValues()))
    return static_cast<int64_t>(std::llround(*fall * timeScalePs));
  return std::nullopt;
}

static std::optional<double>
getOutputSlewFromTimingArc(synth::NLDMArcAttr timingArc, double inputSlew,
                           double outputLoad) {
  if (auto rise = interpolateTable(inputSlew, outputLoad,
                                   timingArc.getRiseTransitionIndex1(),
                                   timingArc.getRiseTransitionIndex2(),
                                   timingArc.getRiseTransitionValues()))
    return rise;
  if (auto fall = interpolateTable(inputSlew, outputLoad,
                                   timingArc.getFallTransitionIndex1(),
                                   timingArc.getFallTransitionIndex2(),
                                   timingArc.getFallTransitionValues()))
    return fall;
  return std::nullopt;
}

static bool decodeCCSPilotWaveform(synth::CCSPilotArcAttr arc, bool preferFall,
                                   SmallVectorImpl<double> &times,
                                   SmallVectorImpl<double> &values) {
  auto decodePair = [&](ArrayAttr t, ArrayAttr v) {
    if (t.empty() || v.empty())
      return false;
    if (!decodeNumericArray(t, times) || !decodeNumericArray(v, values))
      return false;
    return !times.empty() && times.size() == values.size();
  };

  if (preferFall) {
    if (decodePair(arc.getCurrentFallTimes(), arc.getCurrentFallValues()))
      return true;
    if (decodePair(arc.getCurrentRiseTimes(), arc.getCurrentRiseValues()))
      return true;
  } else {
    if (decodePair(arc.getCurrentRiseTimes(), arc.getCurrentRiseValues()))
      return true;
    if (decodePair(arc.getCurrentFallTimes(), arc.getCurrentFallValues()))
      return true;
  }
  return false;
}

static double getCCSPilotLoadStretchFactor(double outputLoad) {
  double clamped = std::max(outputLoad, 0.0);
  return std::max(0.5, 1.0 + 0.5 * (clamped - 0.5));
}

static std::optional<double> interpolateCrossing(ArrayRef<double> times,
                                                 ArrayRef<double> values,
                                                 double target) {
  if (times.empty() || values.empty() || times.size() != values.size())
    return std::nullopt;
  for (size_t i = 1, e = times.size(); i < e; ++i) {
    double v0 = values[i - 1], v1 = values[i];
    double t0 = times[i - 1], t1 = times[i];
    if (v0 == target)
      return t0;
    bool brackets =
        (v0 <= target && target <= v1) || (v1 <= target && target <= v0);
    if (!brackets || v0 == v1)
      continue;
    double alpha = (target - v0) / (v1 - v0);
    return t0 + alpha * (t1 - t0);
  }
  if (values.back() == target)
    return times.back();
  return std::nullopt;
}

static bool shouldUseCCSPilotWaveformDelay(Operation *op) {
  if (!op)
    return false;
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto attr = module->getAttr("synth.ccs.pilot.waveform_delay");
  if (!attr)
    return false;
  if (auto b = dyn_cast<BoolAttr>(attr))
    return b.getValue();
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return i.getInt() != 0;
  if (auto s = dyn_cast<StringAttr>(attr))
    return s.getValue().equals_insensitive("true") || s.getValue() == "1";
  return false;
}

static bool extractCCSPilotThresholdMetrics(const DelayContext &ctx,
                                            const LibertyLibrary *liberty,
                                            bool preferFall, double &t50Ps,
                                            double &slew10to90Ps) {
  if (!liberty || ctx.inputIndex < 0 || ctx.outputIndex < 0)
    return false;
  auto cellName = getMappedCellName(ctx.op);
  if (!cellName)
    return false;

  auto arc = liberty->getTypedCCSPilotArc(
      *cellName, static_cast<unsigned>(ctx.inputIndex),
      static_cast<unsigned>(ctx.outputIndex));
  if (!arc)
    return false;

  SmallVector<double> times, values;
  if (!decodeCCSPilotWaveform(*arc, preferFall, times, values))
    return false;

  double scale = getTimeScalePs(ctx.op);
  double stretch = getCCSPilotLoadStretchFactor(ctx.outputLoad);
  for (double &t : times)
    t = t * scale * stretch;

  auto t50 = interpolateCrossing(times, values, 0.5);
  auto t10 = interpolateCrossing(times, values, 0.1);
  auto t90 = interpolateCrossing(times, values, 0.9);
  if (!t50 || !t10 || !t90)
    return false;

  t50Ps = *t50;
  slew10to90Ps = std::abs(*t90 - *t10);
  return true;
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
          double outputSlew = ctx.inputSlew;
          if (auto slew = getOutputSlewFromTimingArc(*timingArc, ctx.inputSlew,
                                                     ctx.outputLoad))
            outputSlew = *slew;

          if (auto delay =
                  getDelayFromTimingArc(*timingArc, ctx.inputSlew,
                                        ctx.outputLoad, getTimeScalePs(ctx.op)))
            return {*delay, outputSlew};
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

  return {0, ctx.inputSlew};
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
// CCSPilotDelayModel
//===----------------------------------------------------------------------===//

CCSPilotDelayModel::CCSPilotDelayModel() = default;

CCSPilotDelayModel::CCSPilotDelayModel(std::unique_ptr<LibertyLibrary> liberty)
    : nldmDelegate(std::move(liberty)) {}

CCSPilotDelayModel::~CCSPilotDelayModel() = default;

DelayResult CCSPilotDelayModel::computeDelay(const DelayContext &ctx) const {
  auto base = nldmDelegate.computeDelay(ctx);

  double t50Ps = 0.0;
  double slew10to90Ps = 0.0;
  if (extractCCSPilotThresholdMetrics(ctx, nldmDelegate.getLibertyLibrary(),
                                      /*preferFall=*/false, t50Ps,
                                      slew10to90Ps)) {
    base.outputSlew = slew10to90Ps;
    if (shouldUseCCSPilotWaveformDelay(ctx.op))
      base.delay = static_cast<int64_t>(std::llround(t50Ps));
  }

  return base;
}

double CCSPilotDelayModel::getInputCapacitance(const DelayContext &ctx) const {
  return nldmDelegate.getInputCapacitance(ctx);
}

bool CCSPilotDelayModel::computeOutputWaveform(
    const DelayContext &ctx, ArrayRef<WaveformPoint> inputWaveform,
    SmallVectorImpl<WaveformPoint> &outputWaveform) const {
  outputWaveform.clear();

  auto result = computeDelay(ctx);
  double delayPs = static_cast<double>(result.delay);
  double baseTime = inputWaveform.empty() ? 0.0 : inputWaveform.front().time;

  if (auto *liberty = nldmDelegate.getLibertyLibrary()) {
    if (ctx.inputIndex >= 0 && ctx.outputIndex >= 0) {
      if (auto cellName = getMappedCellName(ctx.op)) {
        if (auto arc = liberty->getTypedCCSPilotArc(
                *cellName, static_cast<unsigned>(ctx.inputIndex),
                static_cast<unsigned>(ctx.outputIndex))) {
          SmallVector<double> times, values;
          bool preferFall =
              inputWaveform.size() >= 2 &&
              inputWaveform.back().value < inputWaveform.front().value;
          if (decodeCCSPilotWaveform(*arc, preferFall, times, values)) {
            double scale = getTimeScalePs(ctx.op);
            double stretch = getCCSPilotLoadStretchFactor(ctx.outputLoad);
            for (auto [t, v] : llvm::zip(times, values))
              outputWaveform.push_back(
                  {baseTime + delayPs + t * scale * stretch, v});
            return true;
          }
        }
      }
    }
  }

  double outputSlew = std::max(result.outputSlew, 1e-6);

  outputWaveform.push_back({baseTime + delayPs, 0.0});
  outputWaveform.push_back({baseTime + delayPs + outputSlew, 1.0});
  return true;
}

//===----------------------------------------------------------------------===//
// MixedNLDMCCSPilotDelayModel
//===----------------------------------------------------------------------===//

MixedNLDMCCSPilotDelayModel::MixedNLDMCCSPilotDelayModel(
    std::unique_ptr<LibertyLibrary> nldmLiberty,
    std::unique_ptr<LibertyLibrary> ccsLiberty, llvm::StringSet<> ccsCells)
    : nldmDelegate(std::move(nldmLiberty)), ccsDelegate(std::move(ccsLiberty)),
      ccsPilotCells(std::move(ccsCells)) {}

MixedNLDMCCSPilotDelayModel::~MixedNLDMCCSPilotDelayModel() = default;

bool MixedNLDMCCSPilotDelayModel::useCCSForCell(Operation *op) const {
  auto cellName = getMappedCellName(op);
  return cellName && ccsPilotCells.contains(*cellName);
}

DelayResult
MixedNLDMCCSPilotDelayModel::computeDelay(const DelayContext &ctx) const {
  if (useCCSForCell(ctx.op))
    return ccsDelegate.computeDelay(ctx);
  return nldmDelegate.computeDelay(ctx);
}

double MixedNLDMCCSPilotDelayModel::getInputCapacitance(
    const DelayContext &ctx) const {
  if (useCCSForCell(ctx.op))
    return ccsDelegate.getInputCapacitance(ctx);
  return nldmDelegate.getInputCapacitance(ctx);
}

bool MixedNLDMCCSPilotDelayModel::computeOutputWaveform(
    const DelayContext &ctx, ArrayRef<WaveformPoint> inputWaveform,
    SmallVectorImpl<WaveformPoint> &outputWaveform) const {
  if (!useCCSForCell(ctx.op))
    return false;
  return ccsDelegate.computeOutputWaveform(ctx, inputWaveform, outputWaveform);
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

std::unique_ptr<DelayModel> circt::synth::timing::createCCSPilotDelayModel() {
  return std::make_unique<CCSPilotDelayModel>();
}

std::unique_ptr<DelayModel>
circt::synth::timing::createCCSPilotDelayModel(ModuleOp module) {
  if (!module)
    return std::make_unique<CCSPilotDelayModel>();

  auto libertyOr = LibertyLibrary::fromModule(module);
  if (failed(libertyOr))
    return std::make_unique<CCSPilotDelayModel>();

  return std::make_unique<CCSPilotDelayModel>(
      std::make_unique<LibertyLibrary>(std::move(*libertyOr)));
}

std::unique_ptr<DelayModel>
circt::synth::timing::createMixedNLDMCCSPilotDelayModel(ModuleOp module) {
  if (!module)
    return std::make_unique<NLDMDelayModel>();

  auto nldmLibertyOr = LibertyLibrary::fromModule(module);
  auto ccsLibertyOr = LibertyLibrary::fromModule(module);
  if (failed(nldmLibertyOr) || failed(ccsLibertyOr))
    return std::make_unique<NLDMDelayModel>();

  llvm::StringSet<> ccsCells;
  if (auto cellsAttr = dyn_cast_or_null<ArrayAttr>(
          module->getAttr("synth.ccs.pilot.cells"))) {
    for (auto attr : cellsAttr)
      if (auto cell = dyn_cast<StringAttr>(attr))
        ccsCells.insert(cell.getValue());
  }

  return std::make_unique<MixedNLDMCCSPilotDelayModel>(
      std::make_unique<LibertyLibrary>(std::move(*nldmLibertyOr)),
      std::make_unique<LibertyLibrary>(std::move(*ccsLibertyOr)),
      std::move(ccsCells));
}
