//===- AIGPasses.td - AIG dialect passes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_AIGPASSES_TD
#define CIRCT_DIALECT_AIG_AIGPASSES_TD

include "mlir/Pass/PassBase.td"

def LowerVariadic : Pass<"aig-lower-variadic", "hw::HWModuleOp"> {
  let summary = "Lower variadic AndInverter operations to binary AndInverter";
}

def LowerWordToBits : Pass<"aig-lower-word-to-bits", "hw::HWModuleOp"> {
  let summary = "Lower multi-bit AndInverter to single-bit ones";
  let dependentDialects = ["comb::CombDialect"];
}

def PrintLongestPathAnalysis : Pass<"aig-print-longest-path-analysis", "mlir::ModuleOp"> {
  let summary = "Print longest path analysis results with detailed timing statistics";
  let description = [{
    This pass performs longest path analysis on AIG circuits and outputs detailed
    timing information including:
    - Delay distribution statistics showing timing levels and path counts
    - Critical path details for the top N fanout points
    - Path history with intermediate debug points for detailed analysis

    The analysis considers each AIG and-inverter operation to have unit delay and
    computes maximum delays through combinational paths across module hierarchies.
  }];
  let options = [
    Option<"outputFile", "output-file", "std::string", "\"-\"",
           "Output file for analysis results (use '-' for stdout)">,
    Option<"test", "test", "bool", "false",
           "Emit longest paths as diagnostic remarks for testing">,
    Option<"showTopKPercent", "show-top-k-percent", "int", "5", "The size of the longest paths to show.">,
    Option<"emitJSON", "emit-json", "bool", "false",
           "Output analysis results in JSON format">
  ];
}

class ExternalSolverPass<string name> : Pass<name, "hw::HWModuleOp"> {
  list<Option> baseOptions = [
    Option<"continueOnFailure", "continue-on-failure", "bool", "false",
           "Don't fail even if the AIGER exporter, external solver, or AIGER importer fail">,
  ];

  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect", "circt::seq::SeqDialect"
  ];
}

def AIGERRunner : ExternalSolverPass<"aig-runner"> {
  let summary = "Run external logic solvers on AIGER files";
  let description = [{
    This pass exports the current hardware module to AIGER format, runs an
    external logic solver on the exported file, and imports the optimized
    results back into the MLIR module.

    The solver command arguments must contain placeholder tokens that are
    replaced at runtime:
    - `<inputFile>`: Replaced with the path to the temporary AIGER input file
    - `<outputFile>`: Replaced with the path where the solver should write results.

    ABCRunner is a wrapper around AIGERRunner that uses ABC as the external
    solver. Hence if the user wants to use ABC, they should use ABCRunner instead
    of AIGERRunner.
  }];
  let options = baseOptions # [
    Option<"solverPath", "solver-path", "std::string", "",
           "Path to the external solver executable">,
    ListOption<"solverArgs", "solver-args", "std::string", "">
  ];
}

def ABCRunner : ExternalSolverPass<"abc-runner"> {
  let summary = "Run ABC on AIGER files";
  let description = [{
    This pass runs ABC on AIGER files. It is a wrapper around AIGERRunner that
    uses ABC as the external solver. It runs the following ABC commands:
    - `read <inputFile>`: Read the AIGER file
    - for each command in `abcCommands`, run `-q <command>`
    - `write <outputFile>`: Write the AIGER file
  }];
  let options =  baseOptions # [
    Option<"abcPath", "abc-path", "std::string", "", "Path to the ABC executable">,
    ListOption<"abcCommands", "abc-commands", "std::string", "">,
  ];
}

def Mapper : Pass<"synthesis-tech-mapper", "mlir::ModuleOp"> {
  let summary = "Technology mapping pass for AIG circuits";
  let description = [{
    This pass performs technology mapping on AIG circuits using cut-based algorithms.
    It implements the approach described in "Improvements to Technology Mapping for
    LUT-Based FPGAs" by Mishchenko et al.

    The mapper performs the following steps:
    1. Cut enumeration: Generates all possible cuts for each node in the AIG
    2. Cut selection: Selects the best cuts based on area and delay metrics using priority cuts
    3. Technology mapping: Maps the selected cuts to target HW modules

    Key features:
    - Support for K-LUT mapping (configurable K)
    - Area-oriented and delay-oriented mapping modes
    - Priority cuts algorithm for improved quality of results
    - Support for sequential elements (registers)
    - Hierarchical mapping across module boundaries
    - Custom technology library using hw::HWModuleOp definitions

    The pass operates on AIG circuits represented using aig.and_inv operations
    and produces optimized circuits suitable for FPGA or ASIC implementation.
    Technology library is specified as a list of hw::HWModuleOp names that define
    the available primitives for mapping.
  }];
  let options = [
    Option<"maxCutSize", "max-cut-size", "int", "4",
           "Maximum cut size (K-LUT size) for technology mapping">,
    Option<"maxCutsPerNode", "max-cuts-per-node", "int", "12",
           "Maximum number of cuts to maintain per node">,
    ListOption<"libraryModules", "library-modules", "std::string",
               "List of hw::HWModuleOp names to use as technology library primitives">,
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def GenericLutMapper : Pass<"synthesis-generic-lut-mapper", "mlir::ModuleOp"> {
  let summary = "Map AIG to genernic LUT operations (comb.truth_table)";
  let description = [{
     This transformation maps aig network into generic LUT operation. 
  }];
  let options = [
    Option<"maxLutSize", "max-lut-size", "int", "8",
           "Maximum cut size (K-LUT size) for technology mapping">,
    Option<"maxCutsPerNode", "max-cuts-per-node", "int", "12",
           "Maximum number of cuts to maintain per node">
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
