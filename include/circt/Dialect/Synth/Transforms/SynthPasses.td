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
           "Maximum number of non-trivial cuts to maintain per node">,
    Option<"maxCutInputSize", "max-cut-input-size", "int", "6",
           "Maximum number of cut inputs to consider">,
  ];
}

class CutRewriterPassBase<string name, string op> : Pass<name, op> {
  list<Option> baseOptions = [
    Option<"maxCutsPerRoot", "max-cuts-per-root", "int", "6",
           "Maximum number of non-trivial cuts to maintain per node">,
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
  let statistics = [
    Statistic<"numCutsCreated", "cuts-created",
              "Number of cuts created during cut enumeration">,
    Statistic<"numCutSetsCreated", "cut-sets-created",
              "Number of cut sets created during cut enumeration">,
    Statistic<"numCutsRewritten", "cuts-rewritten",
              "Number of cuts rewritten">,
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
  let options = baseOptions # [
    Option<"maxEntriesPerKey", "max-entries-per-key", "unsigned", "8",
           "Maximum pin-permutation entries kept per (pattern, N-canonical "
           "key); bounds matching work, dropping only pin-diversity "
           "candidates">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def SeqTechMapper : Pass<"synth-seq-tech-mapper", "mlir::ModuleOp"> {
  let summary = "Technology mapping for sequential elements";
  let description = [{
    This pass maps sequential operations (`seq.firreg`, `seq.compreg`,
    `seq.compreg.ce`) to instances of library cells, the standard sequential
    library-mapping equivalent. Library cells are `hw.module`s in the same `ModuleOp` whose
    bodies contain a `synth.ff` state element (as produced by the Liberty
    importer); design flops are single-bit `seq` operations.

    Only rising-edge flops map today, matching the `seq` positive-edge
    convention. Synchronous resets and enables are emulated with external
    `comb.mux` logic when the chosen cell lacks the pin; asynchronous
    resets require a matching cell pin. Cells with extra inputs (such as
    scan pins) have them tied to zero. Among compatible cells, the one
    with the smallest Liberty `area` wins. Unmappable flops (wide,
    initialized, or without a compatible cell) are left untouched.
  }];
  let statistics = [
    Statistic<"numFlopsMapped", "flops-mapped",
              "Number of sequential ops mapped to library cells">,
    Statistic<"numFlopsSkipped", "flops-skipped",
              "Number of sequential ops left unmapped">,
    Statistic<"numFlopsExploded", "flops-exploded",
              "Number of wide sequential ops exploded bit by bit">
  ];
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect",
                             "seq::SeqDialect", "synth::SynthDialect"];
}

def SeqMemMapper : Pass<"synth-seq-mem-mapper", "mlir::ModuleOp"> {
  let summary = "Technology mapping for SRAMs";
  let description = [{
    This pass maps `seq.firmem` memories to instances of SRAM macro cells,
    the standard memory-mapping equivalent for the macro-mapping part. SRAM
    macros are `hw.module.extern` ops carrying a `synth.sram.macro`
    descriptor (geometry plus dedicated read/write ports).

    Only exact matches map today: equal width and depth, equal read/write
    port counts in order, a single shared clock, equal read and write
    latencies, undefined read/write-under-write behavior, no power-on init,
    and no read-write ports. Write masks require a matching macro mask
    width; unmasked writes tie macro masks to all-ones. Design enables
    require macro enable pins and are tied active otherwise. Unmappable
    memories (geometry mismatch, missing features) are left untouched.
    Without a compatible macro, memories fall back to flop arrays with
    address decode (like the standard memory mapper without SRAM libraries):
    single clock, asynchronous read, single-cycle write, full words only,
    at most one write port, no read enables. Single-port strobe macros
    needing read/write arbitration are future work.

    Flop-array fallback is bounded by `max-fallback-bits`: a memory with
    more bits and no macro match is a hard error instead of a silent
    explosion, since depth*width flops plus per-bit address muxes hang
    downstream lowering and mapping (a 16384x64 SRAM becomes ~1M flops
    and ~2M muxes). Provide a matching `synth.sram.macro` extern or
    raise the limit explicitly.
  }];
  let options = [
    Option<"maxFallbackBits", "max-fallback-bits", "unsigned",
           /*default=*/"65536",
           "Maximum memory size in bits allowed to fall back to a flop "
           "array when no SRAM macro matches; larger memories are a "
           "hard error. Set to 0 to forbid all fallback.">
  ];
  let statistics = [
    Statistic<"numMemsMapped", "mems-mapped",
              "Number of memories mapped to SRAM macros">,
    Statistic<"numMemsSkipped", "mems-skipped",
              "Number of memories left unmapped">,
    Statistic<"numMemsExploded", "mems-exploded",
              "Number of memories exploded to flop arrays">
  ];
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect",
                             "seq::SeqDialect", "synth::SynthDialect"];
}

def GenSupergates : Pass<"synth-gen-supergates", "mlir::ModuleOp"> {
  let summary = "Generate supergate library from primitive cells";
  let description = [{
    This pass generates supergates by composing primitive technology library
    cells (including phase-flipped variants with an inverter between the
    inner and outer cell). Supergates are emitted as hw.modules with
    synth.mapping_cost so TechMapper picks them up transparently.

    Only the best-area supergate per exact function is kept, and functions
    already covered by primitive cells are skipped.
  }];
  let options = [
    Option<"maxInputs", "max-inputs", "unsigned", "6",
           "Maximum number of primary inputs per supergate">,
    Option<"allowDuplicateInputs", "allow-duplicate-inputs", "bool", "false",
           "Allow candidates that merge one inner input with one outer bypass input">,
    Option<"maxGates", "max-gates", "unsigned", "3",
           "Maximum number of primitive gates per supergate (phase-flipped\n           variants with an inner inverter need 3)">,
    Option<"maxDelay", "max-delay", "int64_t", "0",
           "Maximum pin-to-pin delay per supergate (0 = unlimited)">,
    Option<"maxCandidatesPerRoot", "max-candidates-per-root", "unsigned",
           "256", "Maximum inner candidates tried per (root,pin)">,
    Option<"maxArea", "max-area", "double", "0.0",
           "Maximum area per supergate (0 = unlimited)">
  ];
  let dependentDialects = ["hw::HWDialect", "circt::synth::SynthDialect",
                             "circt::comb::CombDialect"];
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

def LowerVariadic : Pass<"synth-lower-variadic"> {
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
           "Lower operators with timing information">,
    Option<"reuseSubsets", "reuse-subsets", "bool", /*default=*/"false",
           "Reuse existing logic subsets to minimize area">
  ];
  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect",
    "circt::synth::SynthDialect"
  ];
}

def LowerWordToBits : Pass<"synth-lower-word-to-bits"> {
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
                           "circt::seq::SeqDialect", "circt::verif::VerifDialect"];
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

def StructuralHash : Pass<"synth-structural-hash"> {
  let summary = "Structural hashing (CSE) for Synth operations";
  let description = [{
    This pass performs aggressive structural hashing-based CSE for Synth dialect
    operations, domain-specific to AIG operations to enable operand
    reordering based on structural properties and inversion flag consideration
    for canonicalization.
  }];
}

def MaximumAndCover : Pass<"synth-maximum-and-cover"> {
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

def FunctionalReduction : Pass<"synth-functional-reduction", "hw::HWModuleOp"> {
  let summary = "Functional reduction by simulation-guided candidate merging";
  let description = [{
    This pass performs functional reduction for single-bit `synth` logic. It
    runs random-pattern simulation to build candidate equivalence classes for
    `i1` values with matching signatures, then uses a SAT solver to verify
    each candidate pair before materializing the proven equivalences with
    `synth.choice` operations.

    See "FRAIGs: A Unifying Representation for Logic Synthesis and Verification,
    Tech. Rep., 2005" by Mishchenko et al. for details on the underlying
    algorithm.
  }];

  let options = [
    Option<"testTransformation", "test-transformation", "bool", "false",
           "Enable test transformation mode. In this mode, candidates are "
           "verified only when both values in a pair carry matching "
           "`synth.test.functional_reduction_equivalence_class` attributes.">,
    Option<"numRandomPatterns", "num-random-patterns", "unsigned", "1024",
           "Number of random simulation patterns (must be a positive multiple of 64; "
           "each pattern is one bit in a 64-bit word)">,
    Option<"seed", "seed", "unsigned", "0",
           "Seed for random pattern generation">,
    Option<"satSolver", "sat-solver", "std::string", "\"auto\"",
           "SAT solver backend to use: auto, z3, or cadical">,
    Option<"conflictLimit", "conflict-limit", "int64_t", "100",
           "Per-SAT-call conflict budget. -1 disables the limit.">
  ];
  let statistics = [
    Statistic<"numEquivClasses", "num-equiv-classes",
              "Number of non-trivial equivalence classes after simulation">,
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

def ExactSynthesis : Pass<"synth-exact-synthesis"> {
  let summary = "Rewrite small comb.truth_table ops with exact synthesis";
  let description = [{
    This pass exact-synthesizes small single-bit `comb.truth_table` operations
    into Synth IR using a SAT-based minimum-area search. Unsupported or
    unsolved truth tables are left unchanged. The allowed primitive list
    controls the internal nodes selected by the SAT search. Direct constants,
    projections, and polarity inversions are handled as helper cases; an
    output inversion is materialized as `synth.aig.and_inv` and does not count
    against the area limit.
  }];
  let options = [
    ListOption<"allowedOps", "allowed-ops", "std::string",
               "Allowed BooleanLogicOpInterface primitives, written as "
               "'operation-name:arity' entries such as "
               "'synth.aig.and_inv:2'. May be specified multiple times.">,
    Option<"maxInputs", "max-inputs", "unsigned", "6",
           "Maximum number of truth-table inputs to synthesize">,
    Option<"maxArea", "max-area", "unsigned", "32",
           "Maximum number of internal nodes to search">,
    Option<"satSolver", "sat-solver", "std::string", "\"auto\"",
           "SAT solver backend to use: auto, z3, or cadical">
  ];
  let statistics = [
    Statistic<"numSolved", "solved", "Number of truth tables synthesized">,
    Statistic<"numUnsolved", "unsolved",
              "Number of truth tables left unchanged">
  ];
  let dependentDialects = ["hw::HWDialect", "synth::SynthDialect"];
}

def DeriveMappingCost : Pass<"synth-derive-mapping-cost", "mlir::ModuleOp"> {
  let summary = "Derive TechMapper costs from Liberty NLDM data";
  let description = [{
    This pass derives `synth.mapping_cost` attributes for imported Liberty
    combinational cells from their `synth.timing.nldm` arcs and cell `area`,
    so that `synth-tech-mapper` can use them as technology patterns.

    Per input pin, the integer delay is a typical-delay estimate: the mean
    over the pin's cell-rise and cell-fall NLDM values in nanoseconds,
    scaled to picoseconds. Polarity follows `timing_sense` (`non_unate`
    and mixed senses map to positive; the mapper's cost model currently
    ignores polarity). Sensitivity is zero (carried for future load-aware
    mapping). Input capacitances come from the Liberty pin `capacitance`
    in native units. Only single-output, all-`i1` modules with complete
    arc coverage get costs; sequential, multi-output, and arc-less modules
    are skipped. Cells named by `--dont-use` are marked `synth.dont_use`
    and skipped.
  }];
  let options = [
    ListOption<"dontUse", "dont-use", "std::string",
               "Cell names excluded from mapping (marked synth.dont_use)">
  ];
  let statistics = [
    Statistic<"numCostsDerived", "costs-derived",
              "Number of modules given mapping costs">,
    Statistic<"numCostsSkipped", "costs-skipped",
              "Number of modules skipped">
  ];
  let dependentDialects = ["hw::HWDialect", "synth::SynthDialect"];
}

def StripLibToExtern : Pass<"synth-strip-lib-to-extern", "mlir::ModuleOp"> {
  let summary = "Replace library modules with extern declarations";
  let description = [{
    This pass replaces imported library `hw.module`s (marked
    `synth.liberty.cell` or `synth.dont_use`) with `hw.module.extern`
    declarations carrying the same name and ports. Mapping passes need
    the executable bodies; Verilog emission needs only declarations
    (definitions are never emitted, but instances must resolve). Design
    modules are untouched.
  }];
  let statistics = [
    Statistic<"numStripped", "stripped",
              "Number of library modules replaced by externs">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def DedupInstances : Pass<"synth-dedup-instances", "mlir::ModuleOp"> {
  let summary = "Merge duplicate cell instances after technology mapping";
  let description = [{
    Phase-aware technology mapping materializes one inverter cell per negated
    NPN binding, after the last CSE, so shared signals fan out to many
    exactly-duplicate inverters (typically ~30% of all inverters, ~5% of all
    instances). This pass merges exactly-duplicate `hw.instance` ops within
    each module: same target cell, same operands, same parameters. Only
    pure-combinational instances (all-integer operands and results, at least
    one of each) are eligible, which excludes memories, hierarchical modules
    with aggregate ports, and physical-only cells (filler/decap with no
    inputs). Only leaf-cell targets merge: hierarchical targets (design
    modules, composed supergates) and `synth.sram.macro` targets never
    merge, preserving hierarchy and flop counts exactly. Sequential cells
    are excluded even though mapped clocks are plain i1: targets whose body
    holds state or whose liberty pin metadata marks a clock input never
    merge. Instances carrying an `inner_sym` (possibly referenced by XMR)
    or `doNotPrint` are left alone.
  }];
  let statistics = [
    Statistic<"numMerged", "merged",
              "Number of duplicate instances merged">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def SortTopologically : Pass<"synth-sort-topologically", "mlir::ModuleOp"> {
  let summary = "Sort module bodies into topological (SSA dominance) order";
  let description = [{
    Technology mapping appends cell instances after their users (sequential
    cells are placed before the combinational drivers TechMapper creates
    later), so mapped modules violate SSA dominance order. ExportVerilog
    tolerates any order, but downstream MLIR consumers (notably circt-lec,
    which verifies inputs on load) do not. This pass stably sorts every
    non-extern module body topologically so definitions precede uses.
    Blocks with combinational cycles are ordered with cycle breaks and
    counted separately.
  }];
  let statistics = [
    Statistic<"numSorted", "sorted",
              "Number of module bodies sorted">,
    Statistic<"numUnsortable", "unsortable",
              "Number of module bodies ordered with cycle breaks">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def TieCells : Pass<"synth-tie-cells", "mlir::ModuleOp"> {
  let summary = "Drive constant instance inputs from tie cells";
  let description = [{
    ExportVerilog emits `hw.constant` drivers as Verilog literals, but the
    reference flow uses tie cells (`conb_1` on sky130, `TIEHIx1`/`TIELOx1`
    on ASAP7). This pass reroutes i1-constant operands of `hw.instance` ops
    to shared tie-cell instances: per design module, at most one instance
    per needed polarity. Tie cells are auto-detected liberty cells with no
    inputs whose output pin function is the constant "1"/"0" (smallest
    area wins); without them the pass is a no-op. Library modules
    themselves are never rewritten.
  }];
  let statistics = [
    Statistic<"numRerouted", "rerouted",
              "Number of constant operands rerouted to tie cells">,
    Statistic<"numTieInstances", "tie-instances",
              "Number of tie-cell instances created">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def EmitSDC : Pass<"synth-emit-sdc", "mlir::ModuleOp"> {
  let summary = "Emit SDC timing constraints for a mapped design";
  let description = [{
    Discovers flop clock roots by tracing clock pins back through
    single-input cells and hierarchy to top-level input ports, then emits
    `create_clock` constraints plus default input/output delays. Clocks
    that do not resolve to a top port (comb-generated, tied, or lost in
    hierarchy) are reported as unconstrained comments. Requires
    --top-module-name.
  }];
  let options = [
    Option<"topModuleName", "top-module-name", "std::string", "\"\"",
           "Top module to emit constraints for (required)">,
    Option<"clockPeriodNs", "clock-period-ns", "double", "10.0",
           "Clock period in nanoseconds">,
    Option<"inputDelayNs", "input-delay-ns", "double", "0.0",
           "Default input delay in nanoseconds">,
    Option<"outputDelayNs", "output-delay-ns", "double", "0.0",
           "Default output delay in nanoseconds">,
    Option<"outputFile", "output-file", "std::string", "\"-\"",
           "Output file (\"-\" for stdout)">,
    Option<"clockGateCells", "clock-gate-cells", "std::string",
           "\"EICG_wrapper:out=in\"",
           "Clock gates as Cell:out=in pairs (comma-separated); the gate "
           "output traces to its input for domain attribution">,
    Option<"clockSourceCells", "clock-source-cells", "std::string",
           "\"FixedClock\"",
           "Cell-name substrings treated as on-chip clock sources "
           "(comma-separated); reported for manual constraint">,
  ];
  let statistics = [
    Statistic<"numClocks", "clocks",
              "Number of constrained clocks emitted">,
    Statistic<"numUnconstrained", "unconstrained",
              "Number of flop clocks with no top-port root">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def ExtractBundles : Pass<"synth-extract-bundles", "mlir::ModuleOp"> {
  let summary = "Group shared-support values for joint multi-output mapping";
  let description = [{
    Enumerates small (2-3 input) cuts over the optimized network, classifies
    AND2/XOR2/MAJ3/XOR3 functions under input permutation, and groups cuts
    sharing a leaf set. XOR3+MAJ3 pairs on a triple (full adders) and
    XOR2+AND2 pairs on a pair (half adders) become `synth.create_bundle`
    ops for the technology mapper to consume jointly into multi-output
    cells. Roots already bundled are skipped. Purely additive: discrete
    logic stays in place, so erasing a bundle is a complete fallback.
  }];
  let statistics = [
    Statistic<"numBundles", "bundles",
              "Number of create_bundle ops created">,
    Statistic<"numFullAdders", "full-adders",
              "Number of full-adder (XOR3+MAJ3) bundles">,
    Statistic<"numHalfAdders", "half-adders",
              "Number of half-adder (XOR2+AND2) bundles">
  ];
  let dependentDialects = ["hw::HWDialect",
                             "synth::SynthDialect"];
}

def MapBundles : Pass<"synth-map-bundles", "mlir::ModuleOp"> {
  let summary = "Map bundles to multi-output cells";
  let description = [{
    Consumes `synth.create_bundle` ops produced by extraction: re-derives
    the joint function of each bundle over its shared support (via small
    cut re-enumeration) and matches it against multi-output library cells
    under input/output permutation (positive phases only; the discrete
    phase-aware flow covers the rest). On a match the bundle becomes one
    cell instance (cheapest area wins); otherwise it is erased and the
    discrete logic maps as before. Nothing downstream ever observes a
    bundle.
  }];
  let statistics = [
    Statistic<"numCandidates", "candidates",
              "Number of multi-output library cells considered">,
    Statistic<"numMapped", "mapped",
              "Number of bundles mapped to multi-output cells">,
    Statistic<"numErased", "erased",
              "Number of bundles erased without a cell match">
  ];
  let dependentDialects = ["hw::HWDialect",
                             "synth::SynthDialect"];
}

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_PASSES_TD
