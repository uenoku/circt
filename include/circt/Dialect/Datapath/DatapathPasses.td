//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_PASSES_TD
#define CIRCT_DIALECT_DATAPATH_PASSES_TD

include "mlir/Pass/PassBase.td"

def DatapathReduceDelay : Pass<"datapath-reduce-delay"> {
  let summary = "Reduce datapath delay at a cost of potentially greater area.";
  let description = [{
    This pass performs datapath transformations to reduce the delay of any 
    matching paths. The transformations include logic replication/duplication to 
    operate on carry-save values, potentially increasing area usage.

    With `pressure-margin-levels` at zero or above, rewrites apply only to
    timing-pressured operations: roots carrying an explicit
    `datapath.refine.policy` record (reduce-delay or fastest), roots whose
    results or operands are within the structural slack margin of critical
    (via TimingPressureAnalysis), or operations created downstream of an
    applied rewrite (region inheritance). Otherwise the pass preserves its
    legacy whole-module behavior.
  }];
  let options = [
    Option<"pressureMarginLevels", "pressure-margin-levels", "int", "-1",
           "Structural slack margin in levels for selective rewriting; "
           "negative preserves legacy whole-module behavior">
  ];
  let statistics = [
    Statistic<"numMatchesConsidered", "matches-considered",
              "Number of pattern matches considered">,
    Statistic<"numMatchesApplied", "matches-applied",
              "Number of pattern rewrites applied">,
    Statistic<"numMatchesSkipped", "matches-skipped-unpressured",
              "Number of matches skipped outside pressured regions">
  ];
  let dependentDialects = ["datapath::DatapathDialect"];
}

#endif // CIRCT_DIALECT_DATAPATH_PASSES_TD
