//===- ImportTimingFeedback.cpp - Import STA timing feedback ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingFeedback.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Timing/TimingJSON.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
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

  // Apply one probe requirement at a flop input: tighten an existing
  // probe or insert a new one. Returns the action and final requirement
  // for the report. Shared by the v1 (display-string) and v2 (structured
  // key) import paths.
  std::pair<std::string, int64_t> applyProbe(hw::InstanceOp inst,
                                             unsigned dataOperandIdx,
                                             int64_t required,
                                             OpBuilder &builder) {
    Value driver = inst.getOperand(dataOperandIdx);
    std::string action;
    if (auto probe = driver.getDefiningOp<TimingProbeOp>()) {
      int64_t existing = probe.getRequiredTimeAttr().getValue().getSExtValue();
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
      inst.setOperand(dataOperandIdx, newProbe.getResult());
      ++numProbesImported;
      action = "created";
    }
    return {action, required};
  }

  // Structured v2 import: resolve endpoint keys from the configured top
  // (falling back to the document top), classify failures, and apply
  // probes. Strict mode fails on the first problem; permissive mode
  // warns per record and continues.

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
    // Dispatch on schema: v2 carries structured endpoint keys, v1 and
    // legacy carry display strings (import-only compatibility).
    StringRef contents = file.get()->getBuffer();
    bool isV2 = false;
    if (auto parsed = json::parse(contents)) {
      if (const auto *obj = parsed->getAsObject())
        if (auto schema = obj->getString("schema"))
          isV2 = *schema == synth::timing::kTimingFeedbackV2Schema;
    }
    if (isV2) {
      importV2(top, contents);
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
      auto [action, required] =
          applyProbe(resolved->instance, resolved->dataOperandIdx,
                     record.requiredPs, builder);
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

  // Display form of a structured key for reports (v1 report shape).
  static std::string displayKey(const synth::timing::TimingObjectKey &key) {
    std::string out;
    for (auto part : key.instancePath) {
      if (!out.empty())
        out += "/";
      out += part.getValue().str();
    }
    if (!out.empty())
      out += "/";
    out += key.moduleName.getValue().str();
    out += ":";
    out += key.portName.getValue().str();
    out += "[" + std::to_string(key.bit) + "]";
    return out;
  }

  void importV2(ModuleOp top, StringRef contents) {
    using synth::timing::TimingFeedbackV2;
    TimingFeedbackV2 feedback;
    if (failed(synth::timing::parseTimingFeedbackV2(contents, feedback,
                                                    top.getContext()))) {
      if (strict) {
        top->emitError("malformed v2 timing feedback");
        signalPassFailure();
      } else {
        top->emitWarning("malformed v2 timing feedback (skipped)");
      }
      return;
    }
    // Configured top wins; the document top is the fallback. A mismatch
    // is an error in strict mode, a warning otherwise.
    StringAttr topName = feedback.top;
    if (!topModuleName.empty()) {
      if (feedback.top.getValue() != topModuleName) {
        if (strict) {
          top->emitError("v2 feedback top mismatch: file has '")
              << feedback.top.getValue() << "', option selects '"
              << topModuleName << "'";
          signalPassFailure();
          return;
        }
        top->emitWarning("v2 feedback top mismatch: file has '")
            << feedback.top.getValue() << "', option selects '" << topModuleName
            << "' (using option)";
      }
      topName = StringAttr::get(top.getContext(), topModuleName);
    }
    numFeedbackRecords += (unsigned)feedback.paths.size();
    OpBuilder builder(top.getContext());
    std::string report;
    llvm::raw_string_ostream reportOut(report);
    reportOut << "{\"schema\": "
                 "\"circt.synth.timing-feedback-report.v1\", "
                 "\"resolved\": [";
    bool firstResolved = true;
    std::vector<std::string> unresolved;
    for (const auto &record : feedback.paths) {
      synth::ResolveFailure reason;
      auto resolved = synth::resolveFeedbackEndpointV2(top, topName,
                                                       record.endpoint, reason);
      if (failed(resolved)) {
        ++numFeedbackUnresolved;
        std::string display =
            displayKey(record.endpoint) + ":" +
            std::string(synth::describeResolveFailure(reason));
        if (strict) {
          top->emitError("cannot resolve v2 feedback endpoint '")
              << displayKey(record.endpoint) << "' ("
              << synth::describeResolveFailure(reason) << ")";
          signalPassFailure();
          return;
        }
        top->emitWarning("cannot resolve v2 feedback endpoint '")
            << displayKey(record.endpoint) << "' ("
            << synth::describeResolveFailure(reason) << ", skipped)";
        unresolved.push_back(display);
        continue;
      }
      ++numFeedbackResolved;
      auto [action, required] =
          applyProbe(resolved->instance, resolved->dataOperandIdx,
                     record.required.value, builder);
      if (!firstResolved)
        reportOut << ", ";
      firstResolved = false;
      reportOut << "{\"endpoint\": \"" << displayKey(record.endpoint)
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
