//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes related to Synthesis.
//
//===----------------------------------------------------------------------===//


#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TestPriorityCuts : Pass<"synth-test-priority-cuts", "hw::HWModuleOp"> {
  let summary = "Test priority cuts for synthesis";
  let options = [
    Option<"maxCutsPerRoot", "max-cuts-per-root", "int", "6",
           "Maximum number of cuts to maintain per node">,
    Option<"maxCutInputSize", "max-cut-input-size", "int", "6",
           "Maximum number of cut inputs to consider">,
  ];
}

class CutRewriterPassBase<string name, string op> : Pass<name, op> {
  list<Option> baseOptions = [
    Option<"maxCutsPerRoot", "max-cuts-per-root", "int", "6",
           "Maximum number of cuts to maintain per node">,
    Option<"strategy", "strategy", "synth::OptimizationStrategy",
           /*default=*/"synth::OptimizationStrategyTiming",
           "Optimization strategy (area vs. timing)",
           [{::llvm::cl::values(
             clEnumValN(synth::OptimizationStrategyArea, "area",
                        "Optimize for area"),
             clEnumValN(synth::OptimizationStrategyTiming, "timing",
                        "Optimize for timing")
           )}]>,
    Option<"test", "test", "bool", "false", "Attach timing to IR for testing">
  ];
}

def TechMapper : CutRewriterPassBase<"synth-tech-mapper", "mlir::ModuleOp"> {
  let summary = "Technology mapping using cut rewriting";
  let description = [{
    This pass performs technology mapping by converting logic network
    (AIG etc) representations into technology-specific gate implementations.
    It uses cut-based rewriting with priority cuts and NPN canonical forms for
    efficient pattern matching.

    The pass serves dual purposes: providing practical technology mapping
    capabilities and acting as a test vehicle for the cut rewriting framework,
    since testing cut enumeration and pattern matching algorithms directly
    would otherwise be difficult without a concrete application.

    Supports both area and timing optimization strategies.
  }];
  let options = baseOptions;
  let dependentDialects = ["hw::HWDialect"];
}

def GenericLutMapper : CutRewriterPassBase<"synth-generic-lut-mapper",
                                           "hw::HWModuleOp"> {
  let summary = "LUT mapping using generic K-input LUTs";
  let description = [{
    This pass performs technology mapping using generic K-input lookup tables
    (LUTs). It converts combinational logic networks into implementations
    using K-input LUTs (comb.truth_table) with unit area cost and delay.
  }];
  let options = baseOptions # [
    Option<"maxLutSize", "max-lut-size", "unsigned", /*default=*/"6",
           "Maximum number of inputs per LUT">
  ];
  let dependentDialects = ["comb::CombDialect"];
}

def LowerVariadic : Pass<"synth-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic operations to binary operations";
  let description = [{
    This pass lowers variadic operations to binary operations using a
    delay-aware algorithm. For commutative operations, it builds a balanced
    tree by combining values with the earliest arrival times first to minimize
    the critical path.
  }];
  let options = [
    ListOption<"opNames", "op-names", "std::string",
               "Specify operation names to lower (empty means all)">,
    Option<"timingAware", "timing-aware", "bool", "true",
           "Lower operators with timing information">
  ];
  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect",
    "circt::synth::SynthDialect"
  ];
}

def LowerWordToBits : Pass<"synth-lower-word-to-bits", "hw::HWModuleOp"> {
  let summary = "Lower multi-bit AndInverter to single-bit ones";
  let dependentDialects = ["comb::CombDialect"];
  let statistics = [
    Statistic<"numLoweredBits", "num-lowered-bits",
              "Number of total bits lowered including constant">,
    Statistic<"numLoweredConstants", "num-lowered-constants",
              "Number of total constant bits lowered">,
    Statistic<"numLoweredOps", "num-lowered-ops",
              "Number of total operations lowered">,
  ];
}

def PrintLongestPathAnalysis
    : Pass<"synth-print-longest-path-analysis", "mlir::ModuleOp"> {
  let summary =
      "Print longest path analysis results with detailed timing statistics";
  let description = [{
    This pass performs longest path analysis on AIG circuits and outputs detailed
    timing information including:
    - Delay distribution statistics showing timing levels and path counts
    - Critical path details for the top N end points
    - Path history with intermediate debug points for detailed analysis

    The analysis considers each AIG and-inverter operation to have unit delay and
    computes maximum delays through combinational paths across module hierarchies.
  }];
  let options = [Option<
                     "outputFile", "output-file", "std::string", "\"-\"",
                     "Output file for analysis results (use '-' for stdout)">,
                 Option<"test", "test", "bool", "false",
                        "Emit longest paths as diagnostic remarks for testing">,
                 Option<"showTopKPercent", "show-top-k-percent", "int", "5",
                        "The size of the longest paths to show.">,
                 Option<"emitJSON", "emit-json", "bool", "false",
                        "Output analysis results in JSON format">,
                 Option<"topModuleName", "top-module-name", "std::string", "",
                        "Name of the top module to analyze (empty for automatic "
                        "inference from instance graph)">];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect",
                           "circt::synth::SynthDialect"];
}

def PrintResourceUsageAnalysis
    : Pass<"synth-print-resource-usage-analysis", "mlir::ModuleOp"> {
  let summary = "Print resource usage analysis results";
  let description = [{
    This pass performs resource usage analysis on circuits containing Synth,
    Comb, Seq, and HW dialect operations, and outputs resource utilization
    information.

    The analysis computes both local and total resource usage across module
    hierarchies. Local usage counts resources directly in a module, while total
    usage includes all resources from child module instances.

    JSON format (--emit-json):
      [
        {
          "moduleName": "<module_name>",
          "local": {
            "<operation_name>": <count>,
            ...
          },
          "total": {
            "<operation_name>": <count>,
            ...
          },
          "instances": [
            {
              "instanceName": "<instance_name>",
              "moduleName": "<child_module_name>",
              "usage": {
                "moduleName": "<child_module_name>",
                "local": { ... },
                "total": { ... },
                "instances": [ ... ]
              }
            },
            ...
          ]
        },
        ...
      ]

    The JSON schema is recursive: each module's "instances" array contains
    objects with a "usage" field that has the same structure as the top-level
    module object (moduleName, local, total, instances). This allows the full
    module hierarchy to be represented.

    - "local": Resources used directly in this module (not including children)
    - "total": Resources including all child instances (sum of local + children)
    - "instances": Array of child module instances with their full usage data

    When --top-module-name is empty, the pass analyzes all inferred top-level
    modules and outputs results for each in the top-level JSON array.
  }];
  let options = [Option<"outputFile", "output-file", "std::string", "\"-\"",
                        "Output file for analysis results (use '-' for stdout)">,
                 Option<"emitJSON", "emit-json", "bool", "false",
                        "Output analysis results in JSON format">,
                 Option<"topModuleName", "top-module-name", "std::string", "",
                        "Name of the top module to analyze">];
}

class ExternalSolverPass<string name> : Pass<name, "hw::HWModuleOp"> {
  list<Option> baseOptions = [Option<"continueOnFailure", "continue-on-failure",
                                     "bool", "false",
                                     "Don't fail even if the AIGER exporter, "
                                     "external solver, or AIGER importer fail">,
  ];

  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect",
                           "circt::seq::SeqDialect"];
}

def AIGERRunner : ExternalSolverPass<"synth-aiger-runner"> {
  let summary = "Run external solver on AIGER files";
  let description = [{
    This pass runs an external solver on AIGER files. It exports the current
    module to AIGER format, runs the external solver, and imports the result
    back into the module.
  }];
  let options =
      baseOptions#[Option<"aigerPath", "aiger-path", "std::string", "",
                          "Path to the AIGER file">,
                   Option<"solverPath", "solver-path", "std::string", "",
                          "Path to the external solver">,
                   ListOption<"solverArgs", "solver-args", "std::string", "">,
  ];
}

def ABCRunner : ExternalSolverPass<"synth-abc-runner"> {
  let summary = "Run ABC on AIGER files";
  let description = [{
    This pass runs ABC on AIGER files. It is a wrapper around AIGERRunner that
    uses ABC as the external solver. It runs the following ABC commands:
    - `read <inputFile>`: Read the AIGER file
    - for each command in `abcCommands`, run `-q <command>`
    - `write <outputFile>`: Write the AIGER file
  }];
  let options =
      baseOptions#[Option<"abcPath", "abc-path", "std::string", "",
                          "Path to the ABC executable">,
                   ListOption<"abcCommands", "abc-commands", "std::string", "">,
  ];
}

def StructuralHash : Pass<"synth-structural-hash", "hw::HWModuleOp"> {
  let summary = "Structural hashing (CSE) for Synth operations";
  let description = [{
    This pass performs aggressive structural hashing-based CSE for Synth dialect
    operations (AIG/MIG), domain-specific to AIG/MIG operations to enable operand
    reordering based on structural properties and inversion flag consideration
    for canonicalization.
  }];
}

def MaximumAndCover : Pass<"synth-maximum-and-cover", "hw::HWModuleOp"> {
  let summary = "Maximum And Cover for And-Inverter";
  let description = [{
    This pass performs maximum AND-cover optimization by collapsing single-fanout
    and-inverter ops into their users
  }];
}

def SOPBalancing : CutRewriterPassBase<"synth-sop-balancing", "hw::HWModuleOp"> {
  let summary = "SOP (Sum-of-Products) balancing for delay optimization";
  let description = [{
    This pass optimizes delay by restructuring the logic network using balanced
    sum-of-products expressions. It enumerates cuts, derives their SOP
    representation, and selects the best cut based on delay and area.

    The algorithm is based on "Delay Optimization Using SOP Balancing" by
    Mishchenko et al. (ICCAD 2011).
  }];
  let options =
      baseOptions#[Option<"maxCutInputSize", "max-cut-input-size", "unsigned",
                          /*default=*/"6", "Maximum number of inputs per cut">];
  let dependentDialects = ["synth::SynthDialect", "comb::CombDialect"];
}

def MIGAlgebraicRewriting : Pass<"synth-mig-algebraic-rewriting", "hw::HWModuleOp"> {
  let summary = "MIG algebraic depth rewriting";
  let description = [{
    This pass performs algebraic rewriting on Majority-Inverter Graphs (MIGs) for
    depth optimization using associativity and distributivity rules in majority-of-3
    logic. The pass applies transformations to reduce the critical path delay by
    restructuring majority operations.

    The transformations are based on the algebraic properties of majority functions:
    - Associativity: maj(a, b, maj(c, d, e)) can be rewritten under certain conditions
    - Distributivity: maj(a, b, maj(c, d, e)) = maj(maj(a, b, c), maj(a, b, d), e)

    This implementation uses a DFS strategy that applies depth rewriting once to all
    output cones whose drivers have maximum levels.
  }];
  let options = [
    Option<"allowAreaIncrease", "allow-area-increase", "bool", "true",
           "Allow area increase while optimizing depth">
  ];

  let statistics = [
    Statistic<"numAssociativity", "num-associativity",
              "Number of total rewrites performed">,
    Statistic<"numDistributivity", "num-distributivity",
              "Number of total rewrites performed">,
  ];
}

def FunctionalReduction : Pass<"synth-functional-reduction", "hw::HWModuleOp"> {
  let summary = "FRAIG-based functional reduction using SAT solver";
  let description = [{
    This pass implements FRAIG (Functionally Reduced And-Inverter Graph) 
    optimization using a SAT solver. It identifies and merges functionally
    equivalent nodes through simulation-based candidate detection followed by
    SAT-based verification.

    The algorithm works in three phases:
    1. Random simulation to partition nodes into equivalence classes
    2. SAT-based verification of candidate equivalences using incremental solving
    3. Node merging when equivalence is proven

    Supports both `synth.aig.and_inv` (AndInverterOp) and `synth.mig.maj_inv` 
    (MajorityInverterOp) operations. Only single-bit (i1) values are processed;
    multi-bit operations should be lowered first using `synth-lower-word-to-bits`.

    Uses CaDiCaL SAT solver if CIRCT_CADICAL_ENABLED is set, otherwise falls
    back to the built-in MiniSAT solver.
  }];
  let options = [
    Option<"numRandomPatterns", "num-random-patterns", "unsigned", "4096",
           "Number of random simulation patterns (each pattern is one bit in a 64-bit word)">,
    Option<"satConflictLimit", "sat-conflict-limit", "unsigned", "100000",
           "Conflict limit per SAT solver call (0 = unlimited)">,
    Option<"enableCEXFeedback", "enable-cex-feedback", "bool", "true",
           "Enable counterexample feedback to refine simulation">,
    Option<"seed", "seed", "unsigned", "0",
           "Seed for random pattern generation (0 = use system entropy)">
  ];
  let statistics = [
    Statistic<"numEquivClasses", "num-equiv-classes",
              "Number of non-trivial equivalence classes after simulation">,
    Statistic<"numSATCalls", "num-sat-calls",
              "Number of SAT solver invocations">,
    Statistic<"numProvedEquiv", "num-proved-equiv",
              "Number of equivalences proven by SAT">,
    Statistic<"numDisprovedEquiv", "num-disproved-equiv",
              "Number of equivalences disproved by SAT (CEX found)">,
    Statistic<"numUnknown", "num-unknown",
              "Number of SAT calls hitting resource limit (UNKNOWN)">,
    Statistic<"numMergedNodes", "num-merged-nodes",
              "Number of nodes merged due to functional equivalence">
  ];
  let dependentDialects = ["synth::SynthDialect"];
}

#ifdef CIRCT_MOCKTURTLE_ENABLED 
include "SynthMockturtlePasses.td"
#endif

def DesignProfiler : Pass<"synth-design-profiler", "mlir::ModuleOp"> {
  let summary = "Early PPA estimation and design profiling";
  let description = [{
    Performs design profiling for PPA estimation. Currently supports timing
    analysis with clock domain classification. Paths are categorized as:
    - reg-to-reg: Both endpoints are registers
    - port-to-reg: Input port to register
    - reg-to-port: Register to output port
    - port-to-port: Combinational paths

    Path filtering is supported via glob patterns:
    - `*` matches zero or more characters
    - `?` matches exactly one character
    - `[abc]` matches any character in the set

    When both start and end filters are specified, paths must match both
    (AND semantics). Multiple patterns in each filter use OR semantics.

    Two analysis backends are available:
    - Default (LongestPathAnalysis): Backward tracing from endpoints
    - Two-Stage (TimingAnalysis): Forward arrival propagation + on-demand path
      enumeration, similar to commercial EDA tools

    Output: <reportDir>/<topModuleName>/timing.txt
  }];
  let options = [
    Option<"topModuleName", "top", "std::string", "",
           "Name of the top module to analyze">,
    Option<"reportDir", "report-dir", "std::string", "\"reports\"",
           "Directory for output reports">,
    ListOption<"filterStartPoints", "filter-start", "std::string",
               "Glob patterns to filter paths by start point names">,
    ListOption<"filterEndPoints", "filter-end", "std::string",
               "Glob patterns to filter paths by end point names">,
    Option<"useTwoStageAnalysis", "two-stage", "bool", "false",
           "Use two-stage timing analysis (forward arrival + path enumeration)">
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect",
                           "circt::synth::SynthDialect"];
}

def PrintTimingAnalysis : Pass<"synth-print-timing-analysis", "mlir::ModuleOp"> {
  let summary = "Print timing analysis report";
  let description = [{
    Runs two-stage timing analysis and writes a timing report to
    <reportDir>/<topModuleName>/timing.txt. Optional start/end glob filters
    restrict reported paths.
  }];
  let options = [
    Option<"topModuleName", "top", "std::string", "",
           "Name of the top module to analyze">,
    Option<"reportDir", "report-dir", "std::string", "\"reports\"",
           "Directory for output reports, or '-' for stdout">,
    ListOption<"filterStartPoints", "filter-start", "std::string",
               "Glob patterns to filter paths by start point names">,
    ListOption<"filterEndPoints", "filter-end", "std::string",
               "Glob patterns to filter paths by end point names">,
    Option<"numPaths", "num-paths", "unsigned", "10",
           "Maximum number of paths to print">,
    Option<"showConvergenceTable", "show-convergence-table", "bool",
           "false", "Include per-iteration slew convergence table">,
    Option<"showWaveformDetails", "show-waveform-details", "bool",
           "false", "Include per-arc waveform details when supported">,
    Option<"maxSlewIterations", "max-slew-iterations", "unsigned", "6",
           "Maximum slew convergence iterations">,
    Option<"slewConvergenceEpsilon", "slew-epsilon", "double", "1e-6",
           "Absolute slew convergence threshold">,
    Option<"slewConvergenceRelativeEpsilon", "slew-relative-epsilon", "double", "0.0",
           "Relative slew convergence threshold (0 disables)">,
    Option<"slewHintDamping", "slew-hint-damping", "double", "1.0",
           "Damping factor for iterative slew-hint updates">,
    Option<"adaptiveSlewHintDampingMode", "adaptive-slew-damping-mode", "std::string", "\"disabled\"",
           "Adaptive damping mode: disabled, conservative, aggressive">,
    Option<"enableWaveformCoupledConvergence", "enable-waveform-coupled-convergence", "bool", "true",
           "Enable waveform-coupled slew convergence heuristics for waveform-capable models">
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect",
                           "circt::synth::SynthDialect"];
}

def SynthPreprocess : Pass<"synth-preprocess", "hw::HWModuleOp"> {
  let summary = "Preprocessing pass for synthesis IR";
  let description = [{
    This pass performs preprocessing transformations on synthesis IR:
    1. Converts specific external modules to seq operations:
       - SimpleClockMux_wrapper -> seq.clock_mux
       - EICG_wrapper -> seq.clock_gate
    2. Pattern matches clock divider implementations and converts them to seq.clock_div

    The clock divider pattern matched is:
      %true = hw.constant true
      %clock_div = seq.firreg %0 clock %clock reset async %reset, %true : i1
      %0 = synth.aig.and_inv not %clock_div : i1

    This is converted to:
      %clock_div_out = seq.clock_div %clock by 1
  }];
  let dependentDialects = ["circt::seq::SeqDialect", "circt::hw::HWDialect",
                           "circt::comb::CombDialect"];
}
#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
