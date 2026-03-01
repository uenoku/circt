//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <tuple>

#define DEBUG_TYPE "synth-print-timing-analysis"

using namespace circt;
using namespace synth;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_PRINTTIMINGANALYSIS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

namespace {

static std::optional<double> parseNumericAttr(Attribute attr) {
  if (!attr)
    return std::nullopt;
  if (auto f = dyn_cast<FloatAttr>(attr))
    return f.getValueAsDouble();
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return static_cast<double>(i.getInt());
  if (auto s = dyn_cast<StringAttr>(attr)) {
    double value = 0.0;
    if (!s.getValue().trim().getAsDouble(value))
      return value;
  }
  return std::nullopt;
}

static StringRef getRequestedDelayModel(ModuleOp module) {
  if (!module)
    return "";
  if (auto attr = module->getAttrOfType<StringAttr>("synth.timing.model"))
    return attr.getValue();
  return "";
}

static timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode
parseAdaptiveMode(StringRef mode) {
  if (mode == "conservative")
    return timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::
        Conservative;
  if (mode == "aggressive")
    return timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::
        Aggressive;
  return timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Disabled;
}

static const timing::TimingArc *findArcBetween(const timing::TimingNode *from,
                                               const timing::TimingNode *to) {
  for (auto *arc : from->getFanout())
    if (arc->getTo() == to)
      return arc;
  for (auto *arc : from->getFanout())
    if (arc->getOp())
      return arc;
  return nullptr;
}

static double getNodeOutputLoad(const timing::TimingNode *node,
                                const timing::DelayModel *model,
                                timing::TimingAnalysis &analysis) {
  if (!model)
    return 0.0;
  double slewHint = analysis.getArrivals().getMaxArrivalSlew(node);
  double total = 0.0;
  for (auto *arc : node->getFanout()) {
    if (!arc->getOp())
      continue;
    timing::DelayContext ctx;
    ctx.op = arc->getOp();
    ctx.inputValue = arc->getInputValue();
    ctx.outputValue = arc->getOutputValue();
    ctx.inputIndex = arc->getInputIndex();
    ctx.outputIndex = arc->getOutputIndex();
    ctx.inputSlew = slewHint;
    total += model->getInputCapacitance(ctx);
  }
  return total;
}

static std::optional<double>
interpolateWaveformCrossing(ArrayRef<timing::WaveformPoint> waveform,
                            double target) {
  if (waveform.empty())
    return std::nullopt;
  for (size_t i = 1, e = waveform.size(); i < e; ++i) {
    auto p0 = waveform[i - 1];
    auto p1 = waveform[i];
    if (p0.value == target)
      return p0.time;
    bool brackets = (p0.value <= target && target <= p1.value) ||
                    (p1.value <= target && target <= p0.value);
    if (!brackets || p0.value == p1.value)
      continue;
    double t = (target - p0.value) / (p1.value - p0.value);
    return p0.time + t * (p1.time - p0.time);
  }
  if (waveform.back().value == target)
    return waveform.back().time;
  return std::nullopt;
}

struct PrintTimingAnalysisPass
    : public impl::PrintTimingAnalysisBase<PrintTimingAnalysisPass> {
  using PrintTimingAnalysisBase::PrintTimingAnalysisBase;

  void runOnOperation() override {
    auto module = getOperation();

    if (topModuleName.empty()) {
      module.emitError("top module name must be specified");
      return signalPassFailure();
    }

    auto top = findTopModule(module, topModuleName);
    if (!top)
      return signalPassFailure();

    timing::TimingAnalysisOptions analysisOptions;
    analysisOptions.keepAllArrivals = true;
    analysisOptions.emitSlewConvergenceTable = showConvergenceTable;
    analysisOptions.emitWaveformDetails = showWaveformDetails;
    analysisOptions.maxSlewIterations = maxSlewIterations;
    analysisOptions.slewConvergenceEpsilon = slewConvergenceEpsilon;
    analysisOptions.slewConvergenceRelativeEpsilon =
        slewConvergenceRelativeEpsilon;
    analysisOptions.slewHintDamping = slewHintDamping;
    analysisOptions.adaptiveSlewHintDampingMode =
        parseAdaptiveMode(adaptiveSlewHintDampingMode);

    std::unique_ptr<timing::DelayModel> configuredModel;
    if (module->hasAttr("synth.liberty.library")) {
      auto requestedModel = getRequestedDelayModel(module);
      if (requestedModel == "ccs-pilot")
        configuredModel = timing::createCCSPilotDelayModel(module);
      else if (requestedModel == "mixed-ccs-pilot")
        configuredModel = timing::createMixedNLDMCCSPilotDelayModel(module);
      else
        configuredModel = timing::createNLDMDelayModel(module);
      analysisOptions.delayModel = configuredModel.get();

      if (auto initial = module->getAttr("synth.nldm.default_input_slew"))
        if (auto value = parseNumericAttr(initial))
          analysisOptions.initialSlew = *value;

      if (analysisOptions.initialSlew == 0.0)
        if (auto lib =
                module->getAttrOfType<DictionaryAttr>("synth.liberty.library"))
          if (auto value =
                  parseNumericAttr(lib.get("default_input_transition")))
            analysisOptions.initialSlew = *value;
    }

    auto analysis =
        timing::TimingAnalysis::create(module, topModuleName, analysisOptions);
    if (!analysis || failed(analysis->runFullAnalysis())) {
      top->emitError("failed to run timing analysis");
      return signalPassFailure();
    }

    std::string error;
    std::unique_ptr<llvm::ToolOutputFile> file;
    llvm::raw_ostream *os = nullptr;

    if (reportDir == "-") {
      os = &llvm::outs();
    } else {
      auto reportPath = buildReportPath(top);
      auto ec = llvm::sys::fs::create_directories(
          llvm::sys::path::parent_path(reportPath));
      if (ec) {
        top->emitError("failed to create report directory '")
            << llvm::sys::path::parent_path(reportPath)
            << "': " << ec.message();
        return signalPassFailure();
      }
      file = mlir::openOutputFile(reportPath, &error);
      if (!file) {
        top->emitError(error);
        return signalPassFailure();
      }
      os = &file->os();
    }

    printReport(*analysis, analysisOptions.delayModel, *os);

    if (file)
      file->keep();
  }

private:
  static std::string formatNodeLabel(const timing::TimingNode *node) {
    return node->getName().str() + "[" + std::to_string(node->getBitPos()) +
           "]";
  }

  static int64_t resolvePathDelay(const timing::TimingPath &path,
                                  timing::TimingAnalysis &analysis) {
    auto *sp = path.getStartPoint();
    auto *ep = path.getEndPoint();
    if (!sp || !ep)
      return path.getDelay();

    auto arrivals = analysis.getArrivals().getArrivals(ep->getId());
    for (const auto &arrival : arrivals)
      if (arrival.startPoint == sp->getId())
        return arrival.arrivalTime;
    return path.getDelay();
  }

  static std::string formatDeltaTrend(ArrayRef<double> deltas) {
    std::string text;
    llvm::raw_string_ostream os(text);
    for (size_t i = 0, e = deltas.size(); i < e; ++i) {
      if (i)
        os << ", ";
      os << llvm::format("%.6g", deltas[i]);
    }
    return os.str();
  }

  static StringRef classifyDeltaTrend(ArrayRef<double> deltas) {
    if (deltas.size() < 2)
      return "insufficient";

    bool nonIncreasing = true;
    bool nonDecreasing = true;
    for (size_t i = 1, e = deltas.size(); i < e; ++i) {
      nonIncreasing &= deltas[i] <= deltas[i - 1];
      nonDecreasing &= deltas[i] >= deltas[i - 1];
    }

    if (nonIncreasing && !nonDecreasing)
      return "decreasing";
    if (nonDecreasing && !nonIncreasing)
      return "increasing";

    double minV = deltas.front();
    double maxV = deltas.front();
    for (double v : deltas) {
      minV = std::min(minV, v);
      maxV = std::max(maxV, v);
    }
    if (maxV - minV <= 1e-12)
      return "flat";
    return "oscillating";
  }

  static double getReductionRatio(ArrayRef<double> deltas) {
    if (deltas.empty() || deltas.front() <= 0.0)
      return 0.0;
    return deltas.back() / deltas.front();
  }

  static StringRef getConvergenceAdvice(StringRef trendClass,
                                        double reductionRatio, bool converged) {
    if (converged && reductionRatio <= 0.1)
      return "stable and fast convergence";
    if (converged)
      return "converged; tighten epsilon if more precision is needed";
    if (trendClass == "oscillating" || trendClass == "increasing")
      return "not converged; reduce damping or use aggressive adaptive mode";
    return "not converged; increase max iterations or relax relative epsilon";
  }

  static StringRef formatAdaptiveDampingMode(
      timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode mode) {
    switch (mode) {
    case timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Disabled:
      return "disabled";
    case timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::
        Conservative:
      return "conservative";
    case timing::TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive:
      return "aggressive";
    }
    return "disabled";
  }

  static void printSlewConvergenceTable(timing::TimingAnalysis &analysis,
                                        llvm::raw_ostream &os) {
    auto deltas = analysis.getLastSlewDeltaHistory();
    auto dampings = analysis.getLastSlewDampingHistory();
    if (deltas.empty())
      return;

    os << "--- Slew Convergence ---\n";
    os << "Iter | Max Slew Delta | Applied Damping\n";
    for (size_t i = 0, e = deltas.size(); i < e; ++i)
      os << (i + 1) << " | " << llvm::format("%.6g", deltas[i]) << " | "
         << llvm::format("%.6g", i < dampings.size() ? dampings[i] : 1.0)
         << "\n";
    os << "\n";
  }

  static hw::HWModuleOp findTopModule(mlir::ModuleOp module,
                                      llvm::StringRef topModuleName) {
    auto top = module.lookupSymbol<hw::HWModuleOp>(topModuleName);
    if (!top)
      module.emitError("top module '") << topModuleName << "' not found";
    return top;
  }

  std::string buildReportPath(hw::HWModuleOp top) {
    llvm::SmallString<128> path(reportDir);
    llvm::sys::path::append(path, top.getModuleName(), "timing.txt");
    return std::string(path.str());
  }

  void printWaveformDetails(const timing::TimingPath &path,
                            timing::TimingAnalysis &analysis,
                            const timing::DelayModel *model,
                            llvm::raw_ostream &os) {
    if (!model || !model->usesWaveformPropagation())
      return;

    SmallVector<timing::TimingNode *> nodes;
    nodes.push_back(path.getStartPoint());
    for (auto *node : path.getIntermediateNodes())
      nodes.push_back(node);
    nodes.push_back(path.getEndPoint());
    if (nodes.size() < 2)
      return;

    os << "  Waveform Details:\n";
    for (size_t i = 1, e = nodes.size(); i < e; ++i) {
      auto *from = nodes[i - 1];
      auto *to = nodes[i];
      auto *arc = findArcBetween(from, to);
      auto *arcTo = arc ? arc->getTo() : to;
      if (!arc || !arc->getOp()) {
        os << "    " << formatNodeLabel(from) << " -> " << formatNodeLabel(to)
           << ": unavailable\n";
        continue;
      }

      timing::DelayContext ctx;
      ctx.op = arc->getOp();
      ctx.inputValue = arc->getInputValue();
      ctx.outputValue = arc->getOutputValue();
      ctx.inputIndex = arc->getInputIndex();
      ctx.outputIndex = arc->getOutputIndex();
      ctx.inputSlew = analysis.getArrivals().getMaxArrivalSlew(from);
      ctx.outputLoad = getNodeOutputLoad(from, model, analysis);

      SmallVector<timing::WaveformPoint> inputWaveform = {
          {0.0, 0.0}, {std::max(ctx.inputSlew, 1e-6), 1.0}};
      SmallVector<timing::WaveformPoint> outputWaveform;
      if (!model->computeOutputWaveform(ctx, inputWaveform, outputWaveform) ||
          outputWaveform.empty()) {
        os << "    " << formatNodeLabel(from) << " -> " << formatNodeLabel(to)
           << ": unavailable\n";
        continue;
      }

      os << "    " << formatNodeLabel(from) << " -> " << formatNodeLabel(arcTo)
         << ":";
      for (const auto &pt : outputWaveform)
        os << " (t=" << llvm::format("%.6g", pt.time)
           << ", v=" << llvm::format("%.6g", pt.value) << ")";

      auto t50 = interpolateWaveformCrossing(outputWaveform, 0.5);
      auto t10 = interpolateWaveformCrossing(outputWaveform, 0.1);
      auto t90 = interpolateWaveformCrossing(outputWaveform, 0.9);
      if (t50)
        os << " t50=" << llvm::format("%.6g", *t50);
      if (t10 && t90)
        os << " slew10-90=" << llvm::format("%.6g", std::abs(*t90 - *t10));
      os << "\n";
    }
  }

  void printPathDetail(const timing::TimingPath &path,
                       timing::TimingAnalysis &analysis,
                       const timing::DelayModel *model, llvm::raw_ostream &os,
                       size_t rank) {
    auto *sp = path.getStartPoint();
    auto *ep = path.getEndPoint();

    auto describeStartKind = [](timing::TimingNodeKind kind) {
      return kind == timing::TimingNodeKind::RegisterOutput ? "register output"
                                                            : "input port";
    };
    auto describeEndKind = [](timing::TimingNodeKind kind) {
      return kind == timing::TimingNodeKind::RegisterInput ? "register input"
                                                           : "output port";
    };

    int64_t resolvedDelay = resolvePathDelay(path, analysis);
    os << "Path " << rank << ": delay = " << resolvedDelay;
    os << "  slack = " << analysis.getSlack(ep) << "\n";
    os << "  Startpoint: " << formatNodeLabel(sp) << " ("
       << describeStartKind(sp->getKind()) << ")\n";
    os << "  Endpoint:   " << formatNodeLabel(ep) << " ("
       << describeEndKind(ep->getKind()) << ")\n";

    auto intermediates = path.getIntermediateNodes();
    if (!intermediates.empty()) {
      os << "  Path:\n";
      os << "    " << formatNodeLabel(sp) << "\n";
      for (auto *node : intermediates)
        os << "      -> " << formatNodeLabel(node) << " (arrival "
           << analysis.getArrivalTime(node) << ")\n";
      os << "      -> " << formatNodeLabel(ep) << "\n";
    }
    if (analysis.shouldEmitWaveformDetails())
      printWaveformDetails(path, analysis, model, os);

    os << "\n";
  }

  void printReport(timing::TimingAnalysis &analysis,
                   const timing::DelayModel *model, llvm::raw_ostream &os) {
    SmallVector<timing::TimingPath> paths;
    if (failed(analysis.getPaths(filterStartPoints, filterEndPoints, paths,
                                 /*maxPaths=*/0))) {
      os << "Error: failed to enumerate timing paths.\n";
      return;
    }

    llvm::sort(
        paths, [&](const timing::TimingPath &a, const timing::TimingPath &b) {
          return resolvePathDelay(a, analysis) > resolvePathDelay(b, analysis);
        });

    SmallVector<timing::TimingPath> uniquePaths;
    std::set<std::tuple<std::string, std::string, int64_t>> seen;
    for (auto &path : paths) {
      auto key = std::make_tuple(path.getStartPoint()->getName().str(),
                                 path.getEndPoint()->getName().str(),
                                 resolvePathDelay(path, analysis));
      if (seen.insert(key).second)
        uniquePaths.push_back(path);
    }

    if (numPaths.getValue() > 0 && uniquePaths.size() > numPaths.getValue())
      uniquePaths.resize(numPaths.getValue());

    os << "=== Timing Report ===\n";
    os << "Module: " << analysis.getGraph().getModule().getModuleName() << "\n";
    os << "Delay Model: " << analysis.getGraph().getDelayModelName() << "\n";
    os << "Initial Slew: " << analysis.getConfiguredInitialSlew() << "\n";
    os << "Slew Hint Damping: " << analysis.getConfiguredSlewHintDamping()
       << "\n";
    os << "Adaptive Slew Damping Mode: "
       << formatAdaptiveDampingMode(
              analysis.getConfiguredAdaptiveSlewHintDampingMode())
       << "\n";
    os << "Applied Slew Hint Damping: "
       << analysis.getLastAppliedSlewHintDamping() << "\n";
    os << "Arrival Iterations: " << analysis.getLastArrivalIterations() << "\n";
    os << "Slew Converged: "
       << (analysis.didLastArrivalConverge() ? "yes" : "no") << "\n";
    os << "Max Slew Delta: " << analysis.getLastMaxSlewDelta() << "\n";
    os << "Relative Max Slew Delta: " << analysis.getLastRelativeSlewDelta()
       << "\n";
    os << "Relative Slew Epsilon: "
       << analysis.getConfiguredSlewConvergenceRelativeEpsilon() << "\n";
    os << "Slew Delta Trend: "
       << formatDeltaTrend(analysis.getLastSlewDeltaHistory()) << "\n";
    auto trendClass = classifyDeltaTrend(analysis.getLastSlewDeltaHistory());
    auto reductionRatio = getReductionRatio(analysis.getLastSlewDeltaHistory());
    os << "Slew Trend Class: " << trendClass << "\n";
    os << "Slew Reduction Ratio: " << llvm::format("%.6g", reductionRatio)
       << "\n";
    os << "Slew Advice: "
       << getConvergenceAdvice(trendClass, reductionRatio,
                               analysis.didLastArrivalConverge())
       << "\n";
    os << "Worst Slack: " << analysis.getWorstSlack() << "\n";
    os << "\n";

    if (analysis.shouldEmitSlewConvergenceTable())
      printSlewConvergenceTable(analysis, os);

    os << "--- Critical Paths (Top "
       << (numPaths.getValue() == 0
               ? uniquePaths.size()
               : std::min<size_t>(numPaths.getValue(), uniquePaths.size()))
       << ") ---\n";
    for (size_t i = 0, e = uniquePaths.size(); i < e; ++i)
      printPathDetail(uniquePaths[i], analysis, model, os, i + 1);
  }
};

} // namespace
