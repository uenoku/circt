//===- TimingReport.cpp - Timing Report Generation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <string>

using namespace circt;
using namespace circt::synth::timing;

static std::string formatNodeLabel(const TimingNode *node) {
  return (node->getName() + "[" + std::to_string(node->getBitPos()) + "]")
      .str();
}

static int64_t resolvePathDelay(const TimingPath &path,
                                const ArrivalAnalysis *arrivals) {
  auto *sp = path.getStartPoint();
  auto *ep = path.getEndPoint();
  if (!arrivals || !sp || !ep)
    return path.getDelay();

  auto infos = arrivals->getArrivals(ep->getId());
  for (const auto &info : infos)
    if (info.startPoint == sp->getId())
      return info.arrivalTime;
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
    TimingAnalysisOptions::AdaptiveSlewHintDampingMode mode) {
  switch (mode) {
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Disabled:
    return "disabled";
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Conservative:
    return "conservative";
  case TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive:
    return "aggressive";
  }
  return "disabled";
}

static void printSlewConvergenceTable(const TimingAnalysis &analysis,
                                      llvm::raw_ostream &os) {
  auto deltas = analysis.getLastSlewDeltaHistory();
  auto dampings = analysis.getLastSlewDampingHistory();
  if (deltas.empty())
    return;

  os << "--- Slew Convergence ---\n";
  os << "Iter | Max Slew Delta | Applied Damping\n";
  for (size_t i = 0, e = deltas.size(); i < e; ++i)
    os << (i + 1) << " | " << llvm::format("%.6g", deltas[i]) << " | "
       << llvm::format("%.6g", i < dampings.size() ? dampings[i] : 1.0) << "\n";
  os << "\n";
}

static const TimingArc *findArcBetween(const TimingNode *from,
                                       const TimingNode *to) {
  // Prefer exact match via fanin of destination.
  for (auto *arc : to->getFanin())
    if (arc->getFrom() == from)
      return arc;
  // Fallback: any arc from `from` with an op.
  for (auto *arc : from->getFanout())
    if (arc->getOp())
      return arc;
  return nullptr;
}

/// Return a compact location string for a timing node.
static std::string getNodeLocation(const TimingNode *node) {
  Value v = node->getValue();
  if (!v)
    return "";
  Location loc =
      isa<BlockArgument>(v)
          ? cast<BlockArgument>(v).getOwner()->getParentOp()->getLoc()
          : v.getDefiningOp()->getLoc();
  std::string s;
  llvm::raw_string_ostream os(s);
  loc.print(os);
  return s;
}

static double getNodeOutputLoad(const TimingNode *node, const DelayModel *model,
                                const ArrivalAnalysis *arrivals) {
  if (!model || !arrivals)
    return 0.0;
  double slewHint = arrivals->getMaxArrivalSlew(node);
  double total = 0.0;
  for (auto *arc : node->getFanout()) {
    if (!arc->getOp())
      continue;
    DelayContext ctx;
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
interpolateWaveformCrossing(ArrayRef<WaveformPoint> waveform, double target) {
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

static void printWaveformDetails(const TimingPath &path,
                                 const DelayModel *model,
                                 const ArrivalAnalysis *arrivals,
                                 llvm::raw_ostream &os) {
  if (!model || !arrivals || !model->usesWaveformPropagation())
    return;

  SmallVector<TimingNode *> nodes;
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

    DelayContext ctx;
    ctx.op = arc->getOp();
    ctx.inputValue = arc->getInputValue();
    ctx.outputValue = arc->getOutputValue();
    ctx.inputIndex = arc->getInputIndex();
    ctx.outputIndex = arc->getOutputIndex();
    ctx.inputSlew = arrivals->getMaxArrivalSlew(from);
    ctx.outputLoad = getNodeOutputLoad(from, model, arrivals);

    SmallVector<WaveformPoint> inputWaveform = {
        {0.0, 0.0}, {std::max(ctx.inputSlew, 1e-6), 1.0}};
    SmallVector<WaveformPoint> outputWaveform;
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

void TimingAnalysis::reportTiming(llvm::raw_ostream &os, size_t numPaths) {
  if (!graph) {
    os << "Error: timing graph not built.\n";
    return;
  }

  os << "=== Timing Report ===\n";
  os << "Module: " << module.getModuleName() << "\n";
  os << "Delay Model: " << graph->getDelayModelName() << "\n";
  os << "Initial Slew: " << getConfiguredInitialSlew() << "\n";
  os << "Slew Hint Damping: " << getConfiguredSlewHintDamping() << "\n";
  os << "Adaptive Slew Damping Mode: "
     << formatAdaptiveDampingMode(getConfiguredAdaptiveSlewHintDampingMode())
     << "\n";
  os << "Applied Slew Hint Damping: " << getLastAppliedSlewHintDamping()
     << "\n";
  os << "Arrival Iterations: " << getLastArrivalIterations() << "\n";
  os << "Slew Converged: " << (didLastArrivalConverge() ? "yes" : "no") << "\n";
  os << "Max Slew Delta: " << getLastMaxSlewDelta() << "\n";
  os << "Relative Max Slew Delta: " << getLastRelativeSlewDelta() << "\n";
  os << "Relative Slew Epsilon: "
     << getConfiguredSlewConvergenceRelativeEpsilon() << "\n";
  os << "Effective Relative Slew Epsilon: "
     << getLastEffectiveSlewConvergenceRelativeEpsilon() << "\n";
  os << "Waveform-Coupled Convergence: "
     << (usedWaveformCoupledConvergence() ? "yes" : "no") << "\n";
  auto trendClass = classifyDeltaTrend(getLastSlewDeltaHistory());
  auto reductionRatio = getReductionRatio(getLastSlewDeltaHistory());
  os << "Slew Delta Trend: " << formatDeltaTrend(getLastSlewDeltaHistory())
     << "\n";
  os << "Slew Trend Class: " << trendClass << "\n";
  os << "Slew Reduction Ratio: " << llvm::format("%.6g", reductionRatio)
     << "\n";
  os << "Slew Advice: "
     << getConvergenceAdvice(trendClass, reductionRatio,
                             didLastArrivalConverge())
     << "\n";

  if (requiredTimeAnalysis)
    os << "Worst Slack: " << requiredTimeAnalysis->getWorstSlack() << "\n";

  os << "\n";

  if (shouldEmitSlewConvergenceTable())
    printSlewConvergenceTable(*this, os);

  // Get K worst paths
  SmallVector<TimingPath> paths;
  if (failed(getKWorstPaths(numPaths, paths))) {
    os << "Error: failed to enumerate paths.\n";
    return;
  }

  os << "--- Critical Paths (Top " << numPaths << ") ---\n";
  for (size_t i = 0; i < paths.size(); ++i) {
    auto &path = paths[i];
    int64_t slack = 0;
    if (requiredTimeAnalysis)
      slack = requiredTimeAnalysis->getSlack(path.getEndPoint());

    int64_t resolvedDelay = resolvePathDelay(path, arrivals.get());
    os << "Path " << (i + 1) << ": delay = " << resolvedDelay;
    if (requiredTimeAnalysis)
      os << "  slack = " << slack;
    os << "\n";

    auto *sp = path.getStartPoint();
    auto *ep = path.getEndPoint();
    os << "  Startpoint: " << formatNodeLabel(sp) << " ("
       << (sp->getKind() == TimingNodeKind::RegisterOutput ? "register output"
                                                           : "input port")
       << ")\n";
    os << "  Endpoint:   " << formatNodeLabel(ep) << " ("
       << (ep->getKind() == TimingNodeKind::RegisterInput ? "register input"
                                                          : "output port")
       << ")\n";

    // Print path detail if intermediate nodes are available
    auto intermediates = path.getIntermediateNodes();

    // Build full node list: start -> intermediates -> end.
    SmallVector<TimingNode *> nodes;
    nodes.push_back(sp);
    for (auto *n : intermediates)
      nodes.push_back(n);
    nodes.push_back(ep);

    if (nodes.size() >= 2) {
      constexpr int kPointW = 36;
      constexpr int kDelayW = 10;
      constexpr int kArrW = 10;
      constexpr int kSlewW = 10;

      os << "  Path:\n";
      os << llvm::format("    %-*s %*s %*s %*s  %s\n", kPointW, "Point",
                         kDelayW, "ArcDelay", kArrW, "Arrival", kSlewW, "Slew",
                         "Location");
      os << "    " << std::string(kPointW + kDelayW + kArrW + kSlewW + 20, '-')
         << "\n";

      // Startpoint row.
      {
        int64_t at = arrivals ? arrivals->getMaxArrivalTime(sp) : 0;
        double slew = arrivals ? arrivals->getMaxArrivalSlew(sp) : 0.0;
        std::string loc = getNodeLocation(sp);
        os << llvm::format("    %-*s %*s %*lld %*.4g  %s\n", kPointW,
                           formatNodeLabel(sp).c_str(), kDelayW, "-", kArrW,
                           (long long)at, kSlewW, slew, loc.c_str());
      }

      for (size_t i = 1; i < nodes.size(); ++i) {
        auto *prev = nodes[i - 1];
        auto *cur = nodes[i];
        auto *arc = findArcBetween(prev, cur);

        int64_t arcDelay = arc ? arc->getDelay() : -1;
        int64_t at = arrivals ? arrivals->getMaxArrivalTime(cur) : 0;
        double slew = arrivals ? arrivals->getMaxArrivalSlew(cur) : 0.0;
        std::string loc = getNodeLocation(cur);

        std::string arcDelayStr =
            arcDelay >= 0 ? std::to_string(arcDelay) : "-";
        os << llvm::format("    %-*s %*s %*lld %*.4g  %s\n", kPointW,
                           formatNodeLabel(cur).c_str(), kDelayW,
                           arcDelayStr.c_str(), kArrW, (long long)at, kSlewW,
                           slew, loc.c_str());
      }
    }

    if (shouldEmitWaveformDetails())
      printWaveformDetails(path, options.delayModel, arrivals.get(), os);

    os << "\n";
  }
}
