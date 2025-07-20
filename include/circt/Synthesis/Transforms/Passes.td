//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//


#ifndef CIRCT_SYNTHESIS_TRANSFORMS_PASSES_TD
#define CIRCT_SYNTHESIS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TechMapper : Pass<"synthesis-tech-mapper", "mlir::ModuleOp"> {
  let summary = "";
  let description = [{
  }];
  let options = [
    Option<"maxCutsPerNode", "max-cuts-per-node", "int", "12",
           "Maximum number of cuts to maintain per node">,
    Option<"strategy", "strategy", "synthesis::OptimizationStrategy",
           /*default=*/"synthesis::OptimizationStrategyTiming",
           "Optimization strategy (area vs. timing)",
           [{::llvm::cl::values(
             clEnumValN(synthesis::OptimizationStrategyArea, "area", "Optimize for area"),
             clEnumValN(synthesis::OptimizationStrategyTiming, "timing", "Optimize for timing")
           )}]>
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

def GenericLutMapper : Pass<"synthesis-generic-lut-mapper"> {
  let summary = "Map AIG to genernic LUT operations (comb.truth_table)";
  let description = [{
     This transformation maps aig network into generic LUT operation. 
  }];
  let options = [
    Option<"maxLutSize", "max-lut-size", "int", "6",
           "Maximum cut size (K-LUT size) for technology mapping">,
    Option<"maxCutsPerNode", "max-cuts-per-node", "int", "12",
           "Maximum number of cuts to maintain per node">
  ];
  let dependentDialects = ["circt::comb::CombDialect", "circt::hw::HWDialect"];
}

#endif // CIRCT_TRANSFORMS_PASSES
