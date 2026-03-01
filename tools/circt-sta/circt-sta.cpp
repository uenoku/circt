//===- circt-sta.cpp - CIRCT static timing analysis driver -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;
using namespace synth;

static cl::OptionCategory mainCategory("circt-sta Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::init("-"),
                                          cl::desc("Specify an input file"),
                                          cl::value_desc("filename"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialect",
                              cl::desc("Allow unknown dialects in the input"),
                              cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::opt<std::string> timingReportDir(
    "timing-report-dir",
    cl::desc("Directory for timing report output. "
             "Generates timing report at <dir>/<top>/timing.txt; use '-' for "
             "stdout"),
    cl::init("-"), cl::cat(mainCategory));

static cl::list<std::string> filterStartPoints(
    "filter-start",
    cl::desc("Glob patterns to filter paths by start point names"),
    cl::cat(mainCategory));

static cl::list<std::string> filterEndPoints(
    "filter-end", cl::desc("Glob patterns to filter paths by end point names"),
    cl::cat(mainCategory));

static cl::opt<unsigned> numPaths("num-paths",
                                  cl::desc("Maximum number of paths to print"),
                                  cl::init(10), cl::cat(mainCategory));

static cl::opt<bool> showConvergenceTable(
    "show-convergence-table",
    cl::desc("Include per-iteration slew convergence table in timing report"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> showWaveformDetails(
    "show-waveform-details",
    cl::desc("Include per-arc waveform details in timing report"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<unsigned>
    maxSlewIterations("max-slew-iterations",
                      cl::desc("Maximum slew convergence iterations"),
                      cl::init(6), cl::cat(mainCategory));

static cl::opt<double>
    slewConvergenceEpsilon("slew-epsilon",
                           cl::desc("Absolute slew convergence threshold"),
                           cl::init(1e-6), cl::cat(mainCategory));

static cl::opt<double> slewConvergenceRelativeEpsilon(
    "slew-relative-epsilon",
    cl::desc("Relative slew convergence threshold (0 disables)"), cl::init(0.0),
    cl::cat(mainCategory));

static cl::opt<double>
    slewHintDamping("slew-hint-damping",
                    cl::desc("Damping factor for iterative slew-hint updates"),
                    cl::init(1.0), cl::cat(mainCategory));

static cl::opt<std::string> adaptiveSlewDampingMode(
    "adaptive-slew-damping-mode",
    cl::desc("Adaptive damping mode: disabled, conservative, aggressive"),
    cl::init("disabled"), cl::cat(mainCategory));

static cl::opt<bool> enableWaveformCoupledConvergence(
    "enable-waveform-coupled-convergence",
    cl::desc("Enable waveform-coupled convergence heuristics"), cl::init(true),
    cl::cat(mainCategory));

static LogicalResult executeSTA(MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module)
    return failure();

  if (topName.empty()) {
    llvm::errs() << "error: --top must be specified\n";
    return failure();
  }

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  PrintTimingAnalysisOptions options;
  options.topModuleName = topName;
  options.reportDir = timingReportDir;
  options.numPaths = numPaths;
  options.showConvergenceTable = showConvergenceTable;
  options.showWaveformDetails = showWaveformDetails;
  options.maxSlewIterations = maxSlewIterations;
  options.slewConvergenceEpsilon = slewConvergenceEpsilon;
  options.slewConvergenceRelativeEpsilon = slewConvergenceRelativeEpsilon;
  options.slewHintDamping = slewHintDamping;
  options.adaptiveSlewHintDampingMode = adaptiveSlewDampingMode;
  options.enableWaveformCoupledConvergence = enableWaveformCoupledConvergence;
  for (const auto &pat : filterStartPoints)
    options.filterStartPoints.push_back(pat);
  for (const auto &pat : filterEndPoints)
    options.filterEndPoints.push_back(pat);
  pm.addPass(createPrintTimingAnalysis(options));

  if (failed(pm.run(module.get())))
    return failure();

  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(outputFilename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  module->print(outputFile->os());
  outputFile->keep();
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerAsmPrinterCLOptions();
  circt::synth::registerSynthAnalysisPrerequisitePasses();
  cl::ParseCommandLineOptions(argc, argv,
                              "CIRCT static timing analysis tool\n");

  DialectRegistry registry;
  registry.insert<comb::CombDialect, hw::HWDialect, synth::SynthDialect>();
  MLIRContext context(registry);
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);

  return failed(executeSTA(context));
}
