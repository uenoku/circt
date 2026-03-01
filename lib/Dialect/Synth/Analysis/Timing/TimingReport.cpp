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
  if (deltas.empty())
    return;

  os << "--- Slew Convergence ---\n";
  os << "Iter | Max Slew Delta\n";
  for (size_t i = 0, e = deltas.size(); i < e; ++i)
    os << (i + 1) << " | " << llvm::format("%.6g", deltas[i]) << "\n";
  os << "\n";
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
  os << "Slew Delta Trend: " << formatDeltaTrend(getLastSlewDeltaHistory())
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
    if (!intermediates.empty()) {
      os << "  Path:\n";
      os << llvm::format("    %-30s %8s %8s\n", "Point", "Delay", "Arrival");
      os << "    " << std::string(48, '-') << "\n";
      auto startLabel = formatNodeLabel(sp);
      os << llvm::format("    %-30s %8d %8d\n", startLabel.c_str(), 0, 0);
      // We don't have per-hop delay in the path object, just show node names
      for (auto *node : intermediates) {
        int64_t at = arrivals ? arrivals->getMaxArrivalTime(node) : 0;
        auto nodeLabel = formatNodeLabel(node);
        os << llvm::format("    %-30s %8s %8lld\n", nodeLabel.c_str(), "-",
                           (long long)at);
      }
      auto endLabel = formatNodeLabel(ep);
      os << llvm::format("    %-30s %8s %8lld\n", endLabel.c_str(), "-",
                         (long long)resolvedDelay);
    }
    os << "\n";
  }
}
