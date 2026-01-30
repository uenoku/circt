//===- RequiredTimeAnalysis.cpp - Backward Propagation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/Timing/RequiredTimeAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "required-time-analysis"

using namespace circt;
using namespace circt::synth::timing;

RequiredTimeAnalysis::RequiredTimeAnalysis(const TimingGraph &graph,
                                           const ArrivalAnalysis &arrivals,
                                           Options options)
    : graph(graph), arrivals(arrivals), options(std::move(options)) {
  requiredData.resize(graph.getNumNodes());
}

LogicalResult RequiredTimeAnalysis::run() {
  // Determine RAT at endpoints
  int64_t endpointRAT = options.clockPeriod;
  if (endpointRAT == 0) {
    // Unconstrained: RAT = max arrival time across all endpoints
    for (auto *ep : graph.getEndPoints()) {
      int64_t at = arrivals.getMaxArrivalTime(ep);
      if (at > endpointRAT)
        endpointRAT = at;
    }
  }

  // Initialize RAT at endpoints
  for (auto *ep : graph.getEndPoints()) {
    auto &data = requiredData[ep->getId().index];
    data.setRequiredTime(endpointRAT);
    data.markSet();
  }

  // Backward propagation in reverse topological order
  for (auto *node : graph.getReverseTopologicalOrder()) {
    auto &nodeData = requiredData[node->getId().index];

    // Propagate through fanout arcs backward:
    // For each fanout arc (node -> succ), RAT(node) = min(RAT(succ) - delay)
    for (auto *arc : node->getFanout()) {
      auto *succ = arc->getTo();
      auto &succData = requiredData[succ->getId().index];
      if (!succData.isSet())
        continue;

      int64_t candidateRAT = succData.getRequiredTime() - arc->getDelay();
      if (!nodeData.isSet() || candidateRAT < nodeData.getRequiredTime()) {
        nodeData.setRequiredTime(candidateRAT);
        nodeData.markSet();
      }
    }
  }

  // Compute worst slack across endpoints
  worstSlack = INT64_MAX;
  for (auto *ep : graph.getEndPoints()) {
    int64_t slack = getSlack(ep);
    if (slack < worstSlack)
      worstSlack = slack;
  }
  if (worstSlack == INT64_MAX)
    worstSlack = 0;

  LLVM_DEBUG(llvm::dbgs() << "Required time analysis complete. Worst slack: "
                          << worstSlack << "\n");
  return success();
}

int64_t RequiredTimeAnalysis::getRequiredTime(TimingNodeId nodeId) const {
  if (nodeId.index >= requiredData.size())
    return 0;
  return requiredData[nodeId.index].getRequiredTime();
}

int64_t RequiredTimeAnalysis::getSlack(TimingNodeId nodeId) const {
  return getRequiredTime(nodeId) - arrivals.getMaxArrivalTime(nodeId);
}
