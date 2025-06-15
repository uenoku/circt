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
  let summary = "Print longest path analysis";
  let options = [
    Option<"outputFile", "output-file", "std::string", "\"-\"", "Output file">,
    Option<"test", "test", "bool", "false", "Emit longest paths to the diagnostics.">,
    Option<"showTopKPercent", "show-top-k-percent", "int", "5", "The size of the longest paths to show.">
  ];
}

def AIGERRunner : Pass<"aig-runner", "hw::HWModuleOp"> {
  let summary = "Run external logic solvers on AIGER files";
  let options = [
    Option<"solverPath", "solver-path", "std::string", "", "Path to the external solver executable">,
    ListOption<"solverArgs", "solver-args", "std::string", "">
  ];

  let dependentDialects = [
    "circt::comb::CombDialect", "circt::hw::HWDialect", "circt::aig::AIGDialect", "circt::seq::SeqDialect"
  ];
}

#endif // CIRCT_DIALECT_AIG_AIGPASSES_TD
