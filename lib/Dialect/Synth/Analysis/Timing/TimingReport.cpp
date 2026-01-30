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

using namespace circt;
using namespace circt::synth::timing;

void TimingAnalysis::reportTiming(llvm::raw_ostream &os, size_t numPaths) {
  if (!graph) {
    os << "Error: timing graph not built.\n";
    return;
  }

  os << "=== Timing Report ===\n";
  os << "Module: " << module.getModuleName() << "\n";
  os << "Delay Model: " << graph->getDelayModelName() << "\n";

  if (requiredTimeAnalysis)
    os << "Worst Slack: " << requiredTimeAnalysis->getWorstSlack() << "\n";

  os << "\n";

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

    os << "Path " << (i + 1) << ": delay = " << path.getDelay();
    if (requiredTimeAnalysis)
      os << "  slack = " << slack;
    os << "\n";

    auto *sp = path.getStartPoint();
    auto *ep = path.getEndPoint();
    os << "  Startpoint: " << sp->getName() << " ("
       << (sp->getKind() == TimingNodeKind::RegisterOutput ? "register output"
                                                           : "input port")
       << ")\n";
    os << "  Endpoint:   " << ep->getName() << " ("
       << (ep->getKind() == TimingNodeKind::RegisterInput ? "register input"
                                                          : "output port")
       << ")\n";

    // Print path detail if intermediate nodes are available
    auto intermediates = path.getIntermediateNodes();
    if (!intermediates.empty()) {
      os << "  Path:\n";
      os << llvm::format("    %-30s %8s %8s\n", "Point", "Delay", "Arrival");
      os << "    " << std::string(48, '-') << "\n";
      os << llvm::format("    %-30s %8d %8d\n",
                         sp->getName().str().c_str(), 0, 0);
      // We don't have per-hop delay in the path object, just show node names
      for (auto *node : intermediates) {
        int64_t at = arrivals ? arrivals->getMaxArrivalTime(node) : 0;
        os << llvm::format("    %-30s %8s %8lld\n",
                           node->getName().str().c_str(), "-",
                           (long long)at);
      }
      os << llvm::format("    %-30s %8s %8lld\n",
                         ep->getName().str().c_str(), "-",
                         (long long)path.getDelay());
    }
    os << "\n";
  }
}
