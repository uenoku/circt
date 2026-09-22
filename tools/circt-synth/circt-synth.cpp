//===- circt-synth.cpp - The circt-synth driver -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initializes the 'circt-synth' tool, which performs logic
/// synthesis.
///
//===----------------------------------------------------------------------===//

#include "DatapathRetry.h"
#include "circt/Conversion/ImportAIGER.h"
#include "circt/Conversion/ImportLiberty.h"
#include "circt/Conversion/SynthToComb.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/Datapath/DatapathDialect.h"
#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/Analysis/ResourceUsageAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-synth Options");

/// Allow the user to specify the input file format.  This can be used to
/// override the input, and can be used to specify ambiguous cases like standard
/// input.
enum InputFormatKind { InputUnspecified, InputAIGERFile, InputMLIRFile };

static cl::opt<InputFormatKind> inputFormat(
    "format", cl::desc("Specify the input file format"),
    cl::values(clEnumValN(InputUnspecified, "unspecified",
                          "Unspecified input format"),
               clEnumValN(InputAIGERFile, "aiger", "AIGER input format"),
               clEnumValN(InputMLIRFile, "mlir", "MLIR input format")),
    cl::init(InputUnspecified), cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional, cl::init("-"),
                                          cl::desc("Specify an input file"),
                                          cl::value_desc("filename"),
                                          cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));
static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    useTransformInterpreter("use-transformDialect",
                            cl::desc("transform dialect interpreter"),
                            cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    allowUnregisteredDialects("allow-unregistered-dialect",
                              cl::desc("Allow unknown dialects in the input"),
                              cl::init(false), cl::cat(mainCategory));

enum Until { UntilCombLowering, UntilMapping, UntilEnd };

static auto runUntilValues = llvm::cl::values(
    clEnumValN(UntilCombLowering, "comb-lowering", "Lowering Comb to AIG"),
    clEnumValN(UntilMapping, "mapping", "Run technology/lut mapping"),
    clEnumValN(UntilEnd, "all", "Run entire pipeline (default)"));

static llvm::cl::opt<Until> runUntilBefore(
    "until-before", llvm::cl::desc("Stop pipeline before a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));
static llvm::cl::opt<Until> runUntilAfter(
    "until-after", llvm::cl::desc("Stop pipeline after a specified point"),
    runUntilValues, llvm::cl::init(UntilEnd), llvm::cl::cat(mainCategory));

static cl::opt<bool>
    convertToComb("convert-to-comb",
                  cl::desc("Convert AIG to Comb at the end of the pipeline"),
                  cl::init(false), cl::cat(mainCategory));

static cl::opt<std::string> analysisOutput(
    "analysis-output",
    cl::desc("Directory for analysis output files. When specified, "
             "automatically runs all analyses and writes reports "
             "to this directory. Use '-' for stdout"),
    cl::init(""), cl::cat(mainCategory));

enum AnalysisOutputFormat {
  AnalysisOutputFormatText,
  AnalysisOutputFormatJSON
};

static cl::opt<AnalysisOutputFormat>
    analysisOutputFormat("analysis-output-format",
                         cl::desc("Output format for analysis results"),
                         cl::init(AnalysisOutputFormatText),
                         cl::values(clEnumValN(AnalysisOutputFormatText, "text",
                                               "Text output format"),
                                    clEnumValN(AnalysisOutputFormatJSON, "json",
                                               "JSON output format")),
                         cl::cat(mainCategory));
static cl::opt<int>
    outputLongestPathTopKPercent("output-longest-path-top-k-percent",
                                 cl::desc("Output top K percent of longest "
                                          "paths in the analysis results"),
                                 cl::init(5), cl::cat(mainCategory));

static cl::opt<std::string> topName("top", cl::desc("Top module name"),
                                    cl::value_desc("name"), cl::init(""),
                                    cl::cat(mainCategory));

static cl::list<std::string> abcCommands("abc-commands",
                                         cl::desc("ABC passes to run"),
                                         cl::CommaSeparated,
                                         cl::cat(mainCategory));
static cl::list<std::string>
    libertyFiles("liberty",
                 cl::desc("Liberty files to import as technology libraries"),
                 cl::CommaSeparated, cl::cat(mainCategory));
static cl::list<std::string> sramMacroFiles(
    "sram-macros",
    cl::desc("MLIR files contributing hw.module.extern ops carrying "
             "synth.sram.macro descriptors, merged into the design for "
             "SRAM macro mapping"),
    cl::CommaSeparated, cl::cat(mainCategory));
static cl::list<std::string>
    dontUseCells("dont-use",
                 cl::desc("Cell names excluded from technology mapping"),
                 cl::CommaSeparated, cl::cat(mainCategory));
static cl::opt<bool> stripLibToExtern(
    "strip-lib-to-extern",
    cl::desc("Replace imported library modules with extern declarations "
             "in the output"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> abcPath("abc-path", cl::desc("Path to ABC"),
                                    cl::value_desc("path"), cl::init("abc"),
                                    cl::cat(mainCategory));

static cl::opt<bool>
    ignoreAbcFailures("ignore-abc-failures",
                      cl::desc("Continue on ABC failure instead of aborting"),
                      cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableWordToBits("disable-word-to-bits",
                                       cl::desc("Disable LowerWordToBits pass"),
                                       cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> disableDedup("disable-dedup",
                                  cl::desc("Disable post-map duplicate "
                                           "instance merging"),
                                  cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> loadAwareDelay(
    "load-aware-delay",
    cl::desc("Load-aware arc delays in tech mapping (intrinsic + "
             "sensitivity x estimated output load); helps "
             "selection-dominated logic, can mislead on "
             "broadcast-dominated nets (opt-in)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> bufferHighFanout(
    "buffer-high-fanout",
    cl::desc("Buffer high-fanout nets with fanout-splitting buffers "
             "after mapping (opt-in)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<unsigned> reserveIngredientCuts(
    "reserve-ingredient-cuts",
    cl::desc("Retention slots reserved per node for the smallest cuts "
             "during tech mapping; protects building-block cuts that "
             "combine into larger covers (e.g. majority) from "
             "bottom-up truncation crowding"),
    cl::init(2), cl::cat(mainCategory));
static cl::opt<int>
    maxFanout("max-fanout",
              cl::desc("Maximum per-copy fanout before buffer insertion splits "
                       "a net (with --buffer-high-fanout)"),
              cl::init(32), cl::cat(mainCategory));
static cl::opt<std::string> probeClocks(
    "probe-clocks",
    cl::desc("Comma-separated top-level clock port names for timing "
             "probe derivation right after sequential mapping; empty "
             "derives nothing"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> staViolators(
    "sta-violators",
    cl::desc("STA violator paths file (JSON from sta_scoreboard "
             "--violators-out) for timing-guided buffering: violator "
             "launches buffered (fanout-gated) and violator endpoints "
             "get probes (with --buffer-high-fanout)"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<int> muxBalanceThreshold(
    "mux-balance-threshold",
    cl::desc("Mux-chain depth above which to balance in lowering "
             "(negative keeps legacy 16; powers of two sweep depth "
             "effort)"),
    cl::init(-1), cl::cat(mainCategory));
static cl::opt<bool> forceBooth(
    "force-booth",
    cl::desc("Lower all multiplier partial products to Booth arrays "
             "(datapath structural variant)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> compressToAdd(
    "compress-to-add",
    cl::desc("Lower compressors to variadic adds for downstream "
             "optimization instead of Wallace trees (datapath "
             "structural variant)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<int> datapathRetry(
    "datapath-retry",
    cl::desc("Retry datapath structural variants with STA judging, "
             "keeping the best (0 disables). Variants run in subprocesses "
             "under --retry-timeout; needs --liberty, --top and `sta` "
             "on PATH"),
    cl::init(0), cl::cat(mainCategory));
static cl::opt<unsigned> retryTimeout(
    "retry-timeout",
    cl::desc("Per-candidate wall-clock timeout in seconds for "
             "--datapath-retry (0 disables; exploration hangs without one)"),
    cl::init(600), cl::cat(mainCategory));
static cl::opt<double> retrySlackTarget(
    "retry-slack-target",
    cl::desc("Satisfying slack threshold in ns for --datapath-retry "
             "(min area among satisfying wins)"),
    cl::init(0.0), cl::cat(mainCategory));
static cl::opt<bool> retryKeepTemps(
    "retry-keep-temps",
    cl::desc("Keep --datapath-retry candidate files for inspection"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> timingBackend(
    "timing-backend",
    cl::desc("Timing backend for --datapath-retry judging: sta (external "
             "OpenSTA text report, default) or opentimer (in-process "
             "backend, no Verilog export or sta executable; failures fall "
             "back to sta)"),
    cl::init("sta"), cl::cat(mainCategory));
static cl::opt<std::string> timingResult(
    "timing-result",
    cl::desc("Write the structured timing result JSON to this path "
             "(atomic) for --datapath-retry candidates"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<double> timingClockPeriodNs(
    "timing-clock-period-ns",
    cl::desc("Clock period in ns for --timing-backend=opentimer"),
    cl::init(0.0), cl::cat(mainCategory));
static cl::opt<std::string> timingClockPort(
    "timing-clock-port",
    cl::desc("Top-level clock port name for --timing-backend=opentimer"),
    cl::init("clk"), cl::cat(mainCategory));
static cl::opt<double> timingInputSlewPs(
    "timing-input-slew-ps",
    cl::desc("Default input slew in ps for --timing-backend=opentimer"),
    cl::init(10.0), cl::cat(mainCategory));
static cl::opt<std::string> timingRunOnly(
    "timing-run-only",
    cl::desc("Internal: run the timing backend once over the input MLIR "
             "and write the result JSON to -o (used by --timing-backend "
             "children; not for direct use)"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<bool> datapathRefinement(
    "datapath-refinement",
    cl::desc("Probe-guided selective datapath refinement: derive endpoint "
             "probes early, discover pressured datapath regions, and apply "
             "structural policies only there (opt-in)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> refinementPolicy(
    "datapath-refinement-policy",
    cl::desc("Policy for pressured datapath regions: baseline|reduce-delay|"
             "booth|compress-to-add|fastest|auto (auto is reduce-delay)"),
    cl::init("auto"), cl::cat(mainCategory));
static cl::opt<std::string> datapathPolicyFile(
    "datapath-policy-file",
    cl::desc("JSON policy manifest mapping refinement region IDs to "
             "policies (overrides per-region selection)"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<bool> datapathPolicyStrict(
    "datapath-policy-strict",
    cl::desc("Fail on unknown refinement region IDs or policies in "
             "--datapath-policy-file instead of warning and using baseline"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<int> pressureMarginLevels(
    "datapath-pressure-margin-levels",
    cl::desc("Structural slack margin in levels for refinement selection "
             "(negative disables structural selection)"),
    cl::init(0), cl::cat(mainCategory));
static cl::opt<std::string> timingFeedback(
    "timing-feedback",
    cl::desc("Timing feedback JSON (versioned schema or legacy violator "
             "list) imported as endpoint probes before datapath lowering"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<bool> timingFeedbackStrict(
    "timing-feedback-strict",
    cl::desc("Fail on invalid timing feedback records instead of "
             "diagnosing and skipping them"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> timingFeedbackReport(
    "timing-feedback-report",
    cl::desc("Write the timing feedback resolution report JSON here "
             "(empty disables)"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<std::string> refinementReport(
    "datapath-refinement-report",
    cl::desc("Write the refinement region decision report JSON here "
             "(empty disables)"),
    cl::init(""), cl::cat(mainCategory));
static cl::opt<int> refinementIterations(
    "datapath-refinement-iterations",
    cl::desc("Feedback-guided refinement rounds in --datapath-retry "
             "driver mode (0 disables driver rounds)"),
    cl::init(2), cl::cat(mainCategory));
static cl::opt<double> retryMinImprovementPs(
    "retry-min-improvement-ps",
    cl::desc("Minimum WNS improvement in ps for a refinement round to "
             "continue iterating"),
    cl::init(10.0), cl::cat(mainCategory));
static cl::opt<bool>
    disableDatapath("disable-datapath",
                    cl::desc("Disable datapath optimization passes"),
                    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool>
    disableTimingAware("disable-timing-aware",
                       cl::desc("Disable datapath optimization passes"),
                       cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> enableSOPBalancing("enable-sop-balancing",
                                        cl::desc("Enable SOP balancing"),
                                        cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> restructureMuxTrees(
    "restructure-mux-trees",
    cl::desc("Rephase shared-select mux trees into collapsible 4:1 cones "
             "before technology mapping (opt-in)"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> enableFunctionalReduction(
    "enable-functional-reduction",
    cl::desc("Enable FunctionalReduction during synth optimization"),
    cl::init(false), cl::cat(mainCategory));
static cl::opt<int64_t> functionalReductionConflictLimit(
    "functional-reduction-conflict-limit",
    cl::desc("Per-SAT-call conflict budget for FunctionalReduction. "
             "-1 disables the limit."),
    cl::init(100), cl::cat(mainCategory));

static cl::opt<int> maxCutSizePerRoot("max-cut-size-per-root",
                                      cl::desc("Maximum cut size per root"),
                                      cl::init(8), cl::cat(mainCategory));

static cl::opt<unsigned>
    supergateMaxGates("supergate-max-gates",
                      cl::desc("Maximum primitive gates per supergate"),
                      cl::init(3), cl::cat(mainCategory));
static cl::opt<unsigned>
    maxEntriesPerKey("max-entries-per-key",
                     cl::desc("Maximum pin-permutation entries kept per "
                              "(pattern, N-canonical key) during matching"),
                     cl::init(8), cl::cat(mainCategory));
static cl::opt<std::string> sdcOutput("sdc-output",
                                      cl::desc("Write SDC timing constraints "
                                               "to this file (requires --top)"),
                                      cl::value_desc("filename"), cl::init(""),
                                      cl::cat(mainCategory));
static cl::opt<double> sdcClockPeriod("sdc-clock-period",
                                      cl::desc("Clock period in nanoseconds "
                                               "for emitted SDC"),
                                      cl::init(10.0), cl::cat(mainCategory));
static cl::opt<double> sdcInputDelay("sdc-input-delay",
                                     cl::desc("Default input delay in "
                                              "nanoseconds for emitted SDC"),
                                     cl::init(0.0), cl::cat(mainCategory));
static cl::opt<double> sdcOutputDelay("sdc-output-delay",
                                      cl::desc("Default output delay in "
                                               "nanoseconds for emitted SDC"),
                                      cl::init(0.0), cl::cat(mainCategory));
static cl::opt<std::string> sdcClockGates(
    "sdc-clock-gates",
    cl::desc(
        "Clock gates as Cell:out=in pairs (comma-separated) for emitted SDC"),
    cl::init("EICG_wrapper:out=in"), cl::cat(mainCategory));
static cl::opt<std::string> sdcClockSources(
    "sdc-clock-sources",
    cl::desc("On-chip clock source cell-name substrings (comma-separated) "
             "for emitted SDC"),
    cl::init("FixedClock"), cl::cat(mainCategory));
static cl::opt<unsigned>
    maxMemFallbackBits("max-mem-fallback-bits",
                       cl::desc("Maximum memory size in bits allowed to "
                                "fall back to a flop array when no SRAM "
                                "macro matches; larger memories are a "
                                "hard error"),
                       cl::init(65536), cl::cat(mainCategory));

static cl::opt<bool>
    genSupergates("gen-supergates",
                  cl::desc("Generate supergates (including phase-flipped "
                           "variants) from library cells before technology "
                           "mapping"),
                  cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> extractBundles(
    "extract-bundles",
    cl::desc("Extract shared-support value groups for joint multi-output "
             "cell mapping"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<synth::OptimizationStrategy> synthesisStrategy(
    "synthesis-strategy", cl::desc("Synthesis strategy to use"),
    cl::values(clEnumValN(synth::OptimizationStrategyArea, "area",
                          "Optimize for area"),
               clEnumValN(synth::OptimizationStrategyTiming, "timing",
                          "Optimize for timing")),
    cl::init(synth::OptimizationStrategyTiming), cl::cat(mainCategory));

// Lowering policy (word-level -> bits structure) decoupled from mapping
// selection policy: the two axes span very different structures
// (area lowering is -33% on fifo with better delay), so retrying
// lowering means varying this independently. Defaults to following
// --synthesis-strategy (previous behavior) unless explicitly set.
static cl::opt<synth::OptimizationStrategy> loweringStrategy(
    "lowering-strategy", cl::desc("Datapath lowering policy to use"),
    cl::values(clEnumValN(synth::OptimizationStrategyArea, "area",
                          "Lower for area-lean structures"),
               clEnumValN(synth::OptimizationStrategyTiming, "timing",
                          "Lower for timing-lean structures")),
    cl::init(synth::OptimizationStrategyTiming), cl::cat(mainCategory));

static cl::opt<int>
    lowerToKLUTs("lower-to-k-lut",
                 cl::desc("Lower to generic a truth table op with K inputs"),
                 cl::init(0), cl::cat(mainCategory));

// Opt-in to enable the parameterize constant ports pass.
// NOTE: This is always beneficial for middle-end optimizations but currently
// it's opt-in since it's necessary to run monomorphization (currently not
// exist) before targeting actual synthesis.
static cl::opt<bool>
    enableParameterizeConstantPorts("enable-parameterize-constant-ports",
                                    cl::desc("Enable parameterize constant "
                                             "ports pass"),
                                    cl::init(false), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// SRAM macro import
//===----------------------------------------------------------------------===//

/// Parse an MLIR file and splice its `hw.module.extern` ops carrying a
/// `synth.sram.macro` descriptor into the design module, where SeqMemMapper
/// collects them. A symbol already present in the design keeps the design's
/// own declaration.
static LogicalResult importSRAMMacros(ModuleOp module, StringRef filename,
                                      MLIRContext *context) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  OwningOpRef<ModuleOp> macrosModule =
      parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!macrosModule)
    return failure();
  SymbolTable symbols(module);
  OpBuilder builder(module.getBody(), module.getBody()->end());
  unsigned numImported = 0;
  for (auto externOp : macrosModule->getOps<hw::HWModuleExternOp>()) {
    if (!externOp->hasAttr("synth.sram.macro"))
      continue;
    if (symbols.lookup(externOp.getModuleName()))
      continue;
    (void)symbols.insert(builder.clone(*externOp));
    ++numImported;
  }
  if (numImported == 0)
    llvm::errs() << "warning: no 'synth.sram.macro' externs found in '"
                 << filename << "'\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Main Tool Logic
//===----------------------------------------------------------------------===//

static bool untilReached(Until until) {
  return until >= runUntilBefore || until > runUntilAfter;
}

static void
nestOrAddToHierarchicalRunner(OpPassManager &pm,
                              std::function<void(OpPassManager &pm)> pipeline,
                              const std::string &topName) {
  if (topName.empty()) {
    pipeline(pm.nest<hw::HWModuleOp>());
  } else {
    pm.addPass(circt::createHierarchicalRunner(topName, pipeline));
  }
}

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

// Add a default synthesis pipeline and analysis.
static void populateCIRCTSynthPipeline(PassManager &pm) {
  // Pre-synthesis optimization.
  pm.nest<hw::HWModuleOp>().addPass(hw::createHWBypassInnerSymbols());
  pm.nest<hw::HWModuleOp>().addPass(circt::createSimpleCanonicalizerPass());
  if (enableParameterizeConstantPorts) {
    pm.addPass(hw::createHWParameterizeConstantPorts());
    pm.nest<hw::HWModuleOp>().addPass(circt::createSimpleCanonicalizerPass());
  }
  pm.nest<hw::HWModuleOp>().addPass(mlir::createCSEPass());

  circt::synth::CombLoweringPipelineOptions loweringOptions;
  loweringOptions.disableDatapath = disableDatapath;
  loweringOptions.timingAware = !disableTimingAware;
  loweringOptions.muxBalanceThreshold = muxBalanceThreshold;
  loweringOptions.forceBooth = forceBooth;
  loweringOptions.compressToAdd = compressToAdd;
  // Explicit --lowering-strategy wins; otherwise follow mapping strategy.
  loweringOptions.synthesisStrategy =
      loweringStrategy.getNumOccurrences() > 0 ? loweringStrategy
                                               : synthesisStrategy;
  loweringOptions.datapathRefinement = datapathRefinement;
  loweringOptions.pressureMarginLevels = pressureMarginLevels;

  circt::synth::SynthOptimizationPipelineOptions optimizationOptions;
  optimizationOptions.abcCommands = abcCommands;
  optimizationOptions.abcPath.setValue(abcPath);
  optimizationOptions.ignoreAbcFailures.setValue(ignoreAbcFailures);
  optimizationOptions.disableWordToBits.setValue(disableWordToBits);
  optimizationOptions.timingAware.setValue(!disableTimingAware);
  optimizationOptions.disableSOPBalancing.setValue(!enableSOPBalancing);
  optimizationOptions.disableFunctionalReduction.setValue(
      !enableFunctionalReduction);
  optimizationOptions.functionalReductionConflictLimit.setValue(
      functionalReductionConflictLimit);

  auto pipelineTail = [&](OpPassManager &pm) {
    if (untilReached(UntilCombLowering))
      return;
    circt::synth::buildSynthOptimizationPipeline(pm, optimizationOptions);
    if (untilReached(UntilMapping))
      return;
    if (lowerToKLUTs) {
      circt::synth::GenericLutMapperOptions lutOptions;
      lutOptions.maxLutSize = lowerToKLUTs;
      lutOptions.maxCutsPerRoot = maxCutSizePerRoot;
      pm.addPass(circt::synth::createGenericLutMapper(lutOptions));
    }
  };

  auto pipeline = [&](OpPassManager &pm) {
    circt::synth::buildCombLoweringPipeline(pm, loweringOptions);
    pipelineTail(pm);
  };

  // Probe-guided refinement splits the nested pipeline around a ModuleOp
  // discovery pass: phase A materializes datapath regions, discovery
  // selects policies, phase B lowers with per-region overrides.
  auto refinementPhaseA = [&](OpPassManager &pm) {
    circt::synth::buildCombToDatapathPipeline(pm, loweringOptions);
  };
  auto refinementPhaseB = [&](OpPassManager &pm) {
    circt::synth::buildDatapathLoweringPipeline(pm, loweringOptions);
    pipelineTail(pm);
  };

  if (!untilReached(UntilMapping)) {
    // Derive TechMapper costs from the imported Liberty timing data, and
    // map sequential elements while the IR is still hw/comb/seq:
    // combinational lowering below rewrites library bodies to AIG, which
    // sequential role tracing does not understand.
    synth::DeriveMappingCostOptions mappingCostOptions;
    mappingCostOptions.dontUse.assign(dontUseCells.begin(), dontUseCells.end());
    pm.addPass(synth::createDeriveMappingCost(mappingCostOptions));
    // Memories before flops: flop-array fallbacks from SeqMemMapper are
    // mapped to cells by SeqTechMapper in the same run.
    synth::SeqMemMapperOptions memMapperOptions;
    memMapperOptions.maxFallbackBits = maxMemFallbackBits;
    pm.addPass(synth::createSeqMemMapper(memMapperOptions));
    pm.addPass(synth::createSeqTechMapper());
    // Timing probes (opt-in): derive once on mapped flop D inputs while
    // the original combinational drivers still exist, so pre-map consumers
    // (datapath refinement) and post-map consumers (buffering) share one
    // probe set. Probes ride the lowering/mapping pipeline as transparent
    // pass-throughs; the explicit strip terminator after buffering
    // guarantees none leaks downstream. Default (no --probe-clocks):
    // nothing is added.
    if (!probeClocks.empty()) {
      synth::DeriveProbesOptions probeOptions;
      probeOptions.probeClocks = probeClocks;
      probeOptions.probePeriodNs = sdcClockPeriod;
      pm.addPass(synth::createDeriveProbes(probeOptions));
    }
    // Timing feedback (opt-in): STA endpoint requirements imported as
    // probes on mapped flop D inputs, tightening period-derived probes
    // where both exist. Runs before combinational lowering so datapath
    // refinement can consume it.
    if (!timingFeedback.empty()) {
      synth::ImportTimingFeedbackOptions importOptions;
      importOptions.feedbackFile = timingFeedback;
      importOptions.topModuleName = topName;
      importOptions.strict = timingFeedbackStrict;
      importOptions.feedbackReport = timingFeedbackReport;
      pm.addPass(synth::createImportTimingFeedback(importOptions));
    }
  }

  if (datapathRefinement) {
    nestOrAddToHierarchicalRunner(pm, refinementPhaseA, topName);
    synth::DiscoverRefinementRegionsOptions discoverOptions;
    discoverOptions.pressureMarginLevels = pressureMarginLevels;
    discoverOptions.refinementPolicy = refinementPolicy;
    discoverOptions.policyFile = datapathPolicyFile;
    discoverOptions.policyFileStrict = datapathPolicyStrict;
    discoverOptions.timingFeedback = timingFeedback;
    discoverOptions.topModuleName = topName;
    discoverOptions.timingFeedbackStrict = timingFeedbackStrict;
    discoverOptions.refinementReport = refinementReport;
    pm.addPass(synth::createDiscoverRefinementRegions(discoverOptions));
    nestOrAddToHierarchicalRunner(pm, refinementPhaseB, topName);
  } else {
    nestOrAddToHierarchicalRunner(pm, pipeline, topName);
  }

  if (!untilReached(UntilMapping)) {
    // NOTE: imported library bodies intentionally stay in comb here:
    // with --top, the hierarchical runner above only visits the design
    // hierarchy. TechMapper simulates comb And/Or/Xor directly, so no
    // normalization is needed. (A top-level ConvertCombToSynth would
    // also rewrite design modules outside the --top scope.)
    // Supergates (opt-in) compose library cells before mapping; TechMapper
    // picks generated modules up as ordinary patterns via mapping_cost.
    if (genSupergates) {
      synth::GenSupergatesOptions sgOptions;
      sgOptions.maxGates = supergateMaxGates;
      pm.addPass(synth::createGenSupergates(std::move(sgOptions)));
    }
    // Bundle extraction + joint multi-output mapping (opt-in): propose
    // shared-support pairs, then map each bundle to one cell or erase it.
    // Nothing downstream ever observes a bundle.
    if (extractBundles) {
      pm.addPass(synth::createExtractBundles());
      pm.addPass(synth::createMapBundles());
    }
    // Mux-tree rephasing (opt-in): normalize alternating AIG phases so
    // 4:1 cones present clean mux4 tables to TechMapper (see
    // RestructureMuxTrees pass description). Runs last before mapping.
    if (restructureMuxTrees)
      pm.addPass(synth::createRestructureMuxTrees());
    synth::TechMapperOptions options;
    options.maxCutsPerRoot = maxCutSizePerRoot;
    options.maxEntriesPerKey = maxEntriesPerKey;
    options.strategy = synthesisStrategy;
    options.loadAwareDelay = loadAwareDelay;
    options.reserveIngredientCuts = reserveIngredientCuts;
    pm.addPass(synth::createTechMapper(options));
    // Post-map duplicate merging: TechMapper materializes one inverter per
    // negated binding after the last CSE (~30% of inverters are exact
    // duplicates). Merge before emitting or stripping.
    if (!disableDedup)
      pm.addPass(synth::createDedupInstances());
    // Absorb single-fanout inverters into cheaper complements (NAND for
    // AND+INV, minority for MAJ+INV, etc.). Runs after dedup (fewer INVs)
    // and before stripping (needs library bodies for truth tables).
    if (!disableDedup)
      pm.addPass(synth::createBubblePushing());
    // Tie cells for constant instance inputs (reference parity with
    // Yosys/ORFS netlists); must precede stripping while lib cells are
    // still plain modules.
    pm.addPass(synth::createTieCells());
    // Fanout-splitting buffers for broadcast nets (opt-in): needs plain
    // lib modules (auto-detected buffer cell + clock-pin metadata), so
    // it also precedes stripping. Probes were derived early (above) when
    // clock ports are listed; the buffer pass consumes the ones it finds
    // and the strip terminator below removes the rest, so probes never
    // leak downstream.
    if (bufferHighFanout) {
      synth::BufferHighFanoutOptions bufferOptions;
      bufferOptions.maxFanout = maxFanout;
      bufferOptions.staViolators = staViolators;
      pm.addPass(synth::createBufferHighFanout(bufferOptions));
    }
    // Explicit probe-lifetime terminator. Runs whenever probes could exist
    // (early-derived, imported feedback, or STA-endpoint probes created by
    // guided buffering); a no-op otherwise. Default-off pipelines are
    // untouched.
    if (!probeClocks.empty() || bufferHighFanout || !timingFeedback.empty())
      pm.addPass(synth::createStripTimingProbes());
    if (stripLibToExtern)
      pm.addPass(synth::createStripLibToExtern());
    // NOTE: synth-sort-topologically exists for downstream MLIR consumers
    // needing strict SSA order (e.g. circt-lec), but it is off by default:
    // MLIR's sort is ~5 min single-threaded on ChipTop-scale blocks.
    if (!sdcOutput.empty()) {
      synth::EmitSDCOptions sdcOptions;
      sdcOptions.topModuleName = topName;
      sdcOptions.clockPeriodNs = sdcClockPeriod;
      sdcOptions.inputDelayNs = sdcInputDelay;
      sdcOptions.outputDelayNs = sdcOutputDelay;
      sdcOptions.outputFile = sdcOutput;
      sdcOptions.clockGateCells = sdcClockGates;
      sdcOptions.clockSourceCells = sdcClockSources;
      pm.addPass(synth::createEmitSDC(std::move(sdcOptions)));
    }
  }

  // Run analysis if analysis output is specified.
  if (!analysisOutput.empty()) {
    // Create output directory if it's not stdout
    if (analysisOutput != "-") {
      std::error_code ec = llvm::sys::fs::create_directories(analysisOutput);
      if (ec) {
        llvm::errs() << "Error: cannot create analysis output directory '"
                     << analysisOutput << "': " << ec.message() << "\n";
        return;
      }
    }

    // Determine file extension based on output format
    const char *ext =
        analysisOutputFormat == AnalysisOutputFormatJSON ? ".json" : ".txt";
    auto getFileName = [&](const std::string &base) -> std::string {
      if (analysisOutput == "-")
        return "-";
      llvm::SmallString<128> path(analysisOutput);
      llvm::sys::path::append(path, base + ext);
      return path.str().str();
    };

    // Run longest path analysis
    {
      circt::synth::PrintLongestPathAnalysisOptions options;

      // Construct output path
      options.outputFile = getFileName("longest_path");
      options.showTopKPercent = outputLongestPathTopKPercent;
      options.emitJSON = analysisOutputFormat == AnalysisOutputFormatJSON;
      options.topModuleName = topName;
      pm.addPass(circt::synth::createPrintLongestPathAnalysis(options));
    }

    // Run resource usage analysis
    {
      circt::synth::PrintResourceUsageAnalysisOptions options;
      options.outputFile = getFileName("resource_usage");
      options.emitJSON = analysisOutputFormat == AnalysisOutputFormatJSON;
      options.topModuleName = topName;
      pm.addPass(circt::synth::createPrintResourceUsageAnalysis(options));
    }
  }

  if (convertToComb)
    nestOrAddToHierarchicalRunner(
        pm,
        [&](OpPassManager &pm) {
          pm.addPass(circt::createConvertSynthToComb());
          if (lowerToKLUTs)
            pm.addPass(circt::comb::createLowerComb());
          pm.addPass(createCSEPass());
        },
        topName);
}

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    llvm::errs() << "WARNING: You're attempting to print out a bytecode file.\n"
                    "This is inadvisable as it may cause display problems. If\n"
                    "you REALLY want to taste MLIR bytecode first-hand, you\n"
                    "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

/// Print the operation to the specified stream, emitting bytecode when
/// requested and politely avoiding dumping to terminal unless forced.
static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               mlir::BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeSynthesis(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> input;

  {
    std::string errorMessage;
    input = openInputFile(inputFilename, &errorMessage);
    if (!input) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  }

  // Figure out the input format if unspecified.
  if (inputFormat == InputUnspecified) {
    if (StringRef(inputFilename).ends_with(".aig"))
      inputFormat = InputAIGERFile;
    else if (StringRef(inputFilename).ends_with(".mlir") ||
             StringRef(inputFilename).ends_with(".mlirbc") ||
             mlir::isBytecode(*input))
      inputFormat = InputMLIRFile;
    else {
      llvm::errs() << "unknown input format: "
                      "specify with -format=aiger or -format=mlir\n";
      return failure();
    }
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  OwningOpRef<ModuleOp> module;
  {
    // Parse the provided input files.
    if (inputFormat == InputAIGERFile) {
      circt::aiger::ImportAIGEROptions options;
      module =
          OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
      if (failed(aiger::importAIGER(sourceMgr, &context, ts, module.get(),
                                    &options)))
        module = {};
    } else {
      auto parserTimer = ts.nest("Parse MLIR input");
      module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    }
  }
  if (!module)
    return failure();

  // Merge Liberty technology libraries into the module.
  for (const auto &libertyFile : libertyFiles) {
    if (failed(circt::liberty::importLibertyFile(module.get(), libertyFile)))
      return failure();
  }

  // Merge SRAM macro externs into the module for SeqMemMapper.
  for (const auto &macrosFile : sramMacroFiles) {
    if (failed(importSRAMMacros(module.get(), macrosFile, &context)))
      return failure();
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "circt-synth"));
  if (useTransformInterpreter) {
    pm.addPass(mlir::transform::createInterpreterPass());
  } else {
    populateCIRCTSynthPipeline(pm);
  }

  if (failed(pm.run(module.get())))
    return failure();

  auto timer = ts.nest("Print MLIR output");
  if (failed(printOp(module.get(), outputFile.value()->os())))
    return failure();
  outputFile.value()->keep();
  return success();
}

/// The entry point for the `circt-synth` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeSynthesis` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  circt::synth::registerSynthAnalysisPrerequisitePasses();

  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(argc, argv, "Logic synthesis tool\n\n");

  if (datapathRefinement &&
      (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd)) {
    llvm::errs() << "circt-synth: --datapath-refinement needs full runs "
                    "(no --until-before/after)\n";
    return EXIT_FAILURE;
  }
  if (!timingFeedback.empty() && !datapathRefinement && !bufferHighFanout) {
    llvm::errs() << "circt-synth: warning: --timing-feedback has no consumer "
                    "(enable --datapath-refinement); probes are imported and "
                    "stripped without affecting lowering\n";
  }

  // Register the supported CIRCT dialects and create a context to work
  // with. This precedes the retry driver, which needs a context for
  // timing identity strings even though it exits before synthesis.
  DialectRegistry registry;
  mlir::transform::registerInterpreterPass();

  registry.insert<comb::CombDialect, datapath::DatapathDialect,
                  debug::DebugDialect, emit::EmitDialect, hw::HWDialect,
                  ltl::LTLDialect, om::OMDialect, seq::SeqDialect,
                  sim::SimDialect, synth::SynthDialect, sv::SVDialect,
                  verif::VerifDialect, mlir::transform::TransformDialect>();

  // Register the standard passes we want.
  MLIRContext context(registry);
  if (allowUnregisteredDialects)
    context.allowUnregisteredDialects();

  // Internal timing backend child: run the oracle once and exit.
  if (!timingRunOnly.empty()) {
    synth::DatapathRetryConfig backendConfig;
    backendConfig.top = topName;
    backendConfig.timingClockPeriodNs = timingClockPeriodNs;
    backendConfig.timingClockPort = timingClockPort;
    backendConfig.timingInputSlewPs = timingInputSlewPs;
    std::string liberty;
    for (const auto &lib : libertyFiles) {
      if (!liberty.empty())
        liberty += ",";
      liberty += lib;
    }
    backendConfig.liberty = liberty;
    bool ok = synth::runTimingBackendOnce(backendConfig, &context,
                                           timingRunOnly, outputFilename);
    exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  // In-driver datapath retry: STA-judged structural variants.
  if (datapathRetry > 0) {
    if (runUntilBefore != UntilEnd || runUntilAfter != UntilEnd) {
      llvm::errs() << "circt-synth: --datapath-retry needs full runs "
                      "(no --until-before/after)\n";
      return EXIT_FAILURE;
    }
    synth::DatapathRetryConfig config;
    config.maxVariants = datapathRetry;
    config.slackTarget = retrySlackTarget;
    config.timeoutSecs = retryTimeout;
    config.keepTemps = retryKeepTemps;
    config.refinement = datapathRefinement;
    config.refinementIterations = refinementIterations;
    config.pressureMarginLevels = pressureMarginLevels;
    config.minImprovementPs = retryMinImprovementPs;
    config.initialFeedback = timingFeedback;
    config.selfPath = argv[0];
    for (int i = 0; i < argc; ++i)
      config.argv.push_back(argv[i]);
    std::string liberty;
    for (const auto &lib : libertyFiles) {
      if (!liberty.empty())
        liberty += ",";
      liberty += lib;
    }
    config.liberty = liberty;
    config.top = topName;
    config.timingBackend = timingBackend;
    config.timingResult = timingResult;
    config.timingClockPeriodNs = timingClockPeriodNs;
    config.timingClockPort = timingClockPort;
    config.timingInputSlewPs = timingInputSlewPs;
    config.inputFile = inputFilename;
    config.outputFile = outputFilename;
    config.userSdcOutput = sdcOutput;
    config.mlirContext = &context;
    bool ok = synth::runDatapathRetry(config);
    // Match the exit() discipline below (fast teardown).
    exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
  }

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Context and registry were created above (before the retry driver).

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the synthesis; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeSynthesis(context)));
}
