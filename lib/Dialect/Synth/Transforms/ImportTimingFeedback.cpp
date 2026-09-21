//===- ImportTimingFeedback.cpp - Import STA timing feedback ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingFeedback.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace synth {
#define GEN_PASS_DECL_IMPORTTIMINGFEEDBACK
#define GEN_PASS_DEF_IMPORTTIMINGFEEDBACK
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "synth-import-timing-feedback"

namespace {

struct ImportTimingFeedbackPass
    : public circt::synth::impl::ImportTimingFeedbackBase<
          ImportTimingFeedbackPass> {
  using ImportTimingFeedbackBase<
      ImportTimingFeedbackPass>::ImportTimingFeedbackBase;

  void runOnOperation() override {
    ModuleOp top = getOperation();
    if (feedbackFile.empty())
      return markAllAnalysesPreserved();

    auto file = MemoryBuffer::getFile(feedbackFile);
    if (!file) {
      top->emitError("cannot read timing feedback file '")
          << feedbackFile << "'";
      signalPassFailure();
      return;
    }
    auto feedback = TimingFeedback::parse(
        file.get()->getBuffer(), topModuleName, strict, [&](const Twine &msg) {
          if (strict)
            top->emitError(msg);
          else
            top->emitWarning(msg);
        });
    if (failed(feedback)) {
      signalPassFailure();
      return;
    }

    numFeedbackRecords += (unsigned)feedback->paths.size();
    numFeedbackInvalid += feedback->numInvalid;

    OpBuilder builder(top.getContext());
    std::string report;
    llvm::raw_string_ostream reportOut(report);
    reportOut << "{\"schema\": "
                 "\"circt.synth.timing-feedback-report.v1\", "
                 "\"resolved\": [";
    bool firstResolved = true;
    std::vector<std::string> unresolved;

    for (const auto &record : feedback->paths) {
      auto resolved = resolveFeedbackEndpoint(top, record.endpoint);
      if (failed(resolved)) {
        ++numFeedbackUnresolved;
        unresolved.push_back(record.endpoint);
        continue;
      }
      ++numFeedbackResolved;
      hw::InstanceOp inst = resolved->instance;
      Value driver = inst.getOperand(resolved->dataOperandIdx);
      std::string action;
      int64_t required = record.requiredPs;
      if (auto probe = driver.getDefiningOp<TimingProbeOp>()) {
        int64_t existing =
            probe.getRequiredTimeAttr().getValue().getSExtValue();
        if (required < existing) {
          probe->setAttr("required_time", builder.getI64IntegerAttr(required));
          ++numProbesUpdated;
          action = "updated";
        } else {
          action = "kept";
        }
        required = std::min(required, existing);
      } else {
        builder.setInsertionPoint(inst);
        auto newProbe =
            TimingProbeOp::create(builder, inst.getLoc(), driver, required);
        inst.setOperand(resolved->dataOperandIdx, newProbe.getResult());
        ++numProbesImported;
        action = "created";
      }
      if (!firstResolved)
        reportOut << ", ";
      firstResolved = false;
      reportOut << "{\"endpoint\": \"" << record.endpoint
                << "\", \"required_ps\": " << required << ", \"action\": \""
                << action << "\"}";
    }
    reportOut << "], \"unresolved\": [";
    bool firstUnresolved = true;
    for (const auto &endpoint : unresolved) {
      if (!firstUnresolved)
        reportOut << ", ";
      firstUnresolved = false;
      reportOut << "\"" << endpoint << "\"";
    }
    reportOut << "]}";

    if (!feedbackReport.empty()) {
      std::error_code ec;
      raw_fd_ostream out(feedbackReport, ec);
      if (ec) {
        top->emitError("cannot write feedback report '")
            << feedbackReport << "': " << ec.message();
        signalPassFailure();
        return;
      }
      out << reportOut.str();
    }
  }
};

} // namespace
