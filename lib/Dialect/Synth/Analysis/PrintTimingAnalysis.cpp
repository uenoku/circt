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
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
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

    std::unique_ptr<timing::DelayModel> nldmModel;
    if (module->hasAttr("synth.liberty.library")) {
      nldmModel = timing::createNLDMDelayModel(module);
      analysisOptions.delayModel = nldmModel.get();

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

    printReport(*analysis, *os);

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

  void printPathDetail(const timing::TimingPath &path,
                       timing::TimingAnalysis &analysis, llvm::raw_ostream &os,
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
    os << "\n";
  }

  void printReport(timing::TimingAnalysis &analysis, llvm::raw_ostream &os) {
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
    os << "Arrival Iterations: " << analysis.getLastArrivalIterations() << "\n";
    os << "Slew Converged: "
       << (analysis.didLastArrivalConverge() ? "yes" : "no") << "\n";
    os << "Worst Slack: " << analysis.getWorstSlack() << "\n";
    os << "\n";
    os << "--- Critical Paths (Top "
       << (numPaths.getValue() == 0
               ? uniquePaths.size()
               : std::min<size_t>(numPaths.getValue(), uniquePaths.size()))
       << ") ---\n";
    for (size_t i = 0, e = uniquePaths.size(); i < e; ++i)
      printPathDetail(uniquePaths[i], analysis, os, i + 1);
  }
};

} // namespace
