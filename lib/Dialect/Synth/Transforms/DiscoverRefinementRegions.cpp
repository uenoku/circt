//===- DiscoverRefinementRegions.cpp - Refinement region selection ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/RefinementRegions.h"
#include "circt/Dialect/Synth/Analysis/TimingFeedback.h"
#include "circt/Dialect/Synth/Analysis/TimingPressureAnalysis.h"
#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace synth {
#define GEN_PASS_DECL_DISCOVERREFINEMENTREGIONS
#define GEN_PASS_DEF_DISCOVERREFINEMENTREGIONS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "synth-discover-refinement-regions"

namespace {

struct DiscoverRefinementRegionsPass
    : public circt::synth::impl::DiscoverRefinementRegionsBase<
          DiscoverRefinementRegionsPass> {
  using DiscoverRefinementRegionsBase<
      DiscoverRefinementRegionsPass>::DiscoverRefinementRegionsBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();

    auto defaultPolicy = parseRefinementPolicy(refinementPolicy);
    if (::failed(defaultPolicy)) {
      top->emitError("unknown --refinement-policy '")
          << refinementPolicy << "'";
      signalPassFailure();
      return;
    }

    // Policy manifest (optional).
    std::map<std::string, DatapathRefinementPolicy> manifest;
    if (!policyFile.empty()) {
      auto file = MemoryBuffer::getFile(policyFile);
      if (!file) {
        top->emitError("cannot read policy file '") << policyFile << "'";
        signalPassFailure();
        return;
      }
      auto parsed =
          parsePolicyManifest(file.get()->getBuffer(), policyFileStrict,
                              [&](const Twine &msg) { top->emitWarning(msg); });
      if (::failed(parsed)) {
        if (policyFileStrict)
          top->emitError("invalid policy file '") << policyFile << "'";
        else
          top->emitWarning("ignoring invalid policy file entries");
        signalPassFailure();
        return;
      }
      manifest = std::move(*parsed);
    }

    // Timing feedback overlay (optional). Accepts v1 display records
    // and v2 structured keys (converted to display records; keys from
    // the text adapter round-trip exactly).
    std::vector<TimingFeedbackPath> feedback;
    if (!timingFeedback.empty()) {
      auto file = MemoryBuffer::getFile(timingFeedback);
      if (!file) {
        top->emitError("cannot read timing feedback file '")
            << timingFeedback << "'";
        signalPassFailure();
        return;
      }
      StringRef contents = file.get()->getBuffer();
      bool isV2 = false;
      if (auto parsedJson = json::parse(contents)) {
        if (const auto *obj = parsedJson->getAsObject())
          if (auto schema = obj->getString("schema"))
            isV2 = *schema == timing::kTimingFeedbackV2Schema;
      }
      if (isV2) {
        timing::TimingFeedbackV2 feedbackV2;
        if (failed(timing::parseTimingFeedbackV2(contents, feedbackV2,
                                                 top.getContext()))) {
          if (timingFeedbackStrict) {
            top->emitError("malformed v2 timing feedback");
            signalPassFailure();
            return;
          }
          top->emitWarning("malformed v2 timing feedback (skipped)");
        } else {
          feedback = toV1Paths(feedbackV2);
        }
      } else {
        auto parsed = TimingFeedback::parse(
            contents, topModuleName, timingFeedbackStrict,
            [&](const Twine &msg) { top->emitWarning(msg); });
        if (::failed(parsed)) {
          signalPassFailure();
          return;
        }
        feedback = std::move(parsed->paths);
      }
    }

    TimingPressureAnalysis analysis(top);
    RefinementConfig config;
    config.pressureMarginLevels = pressureMarginLevels;
    config.defaultPolicy = *defaultPolicy;
    config.manifest = std::move(manifest);
    config.feedback = std::move(feedback);
    config.manifestStrict = policyFileStrict;

    RefinementPlan plan =
        discoverRefinementRegions(top, analysis, config, [&](const Twine &msg) {
          top->emitWarning(msg);
        });
    if (!plan.unknownManifestIds.empty() && policyFileStrict) {
      signalPassFailure();
      return;
    }

    unsigned numAnnotated = recordRefinementPlan(plan);

    if (!refinementReport.empty()) {
      std::string report = printRefinementReport(plan, analysis);
      std::error_code ec;
      raw_fd_ostream out(refinementReport, ec);
      if (ec) {
        top->emitError("cannot write refinement report '")
            << refinementReport << "': " << ec.message();
        signalPassFailure();
        return;
      }
      out << report;
    }

    // Statistics.
    for (const auto &region : plan.regions) {
      ++numRegions;
      if (!region.probes.empty())
        ++numRegionsReachingProbes;
      if (region.violated)
        ++numRegionsViolated;
      if (region.policy != DatapathRefinementPolicy::Baseline)
        ++numRegionsRefined;
    }
    numOpsAnnotated += numAnnotated;
    numManifestUnknown += (unsigned)plan.unknownManifestIds.size();
    numFeedbackUnresolved += (unsigned)plan.unresolvedEndpoints.size();
  }
};

} // namespace
