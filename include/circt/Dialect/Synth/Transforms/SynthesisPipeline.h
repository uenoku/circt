//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the default synthesis pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H
#define CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H

#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// Pipeline Options
//===----------------------------------------------------------------------===//
namespace circt {
namespace synth {

enum TargetIR {
  // Lower to And-Inverter Graph
  AIG,
  // Lower to Majority-Inverter Graph
  MIG
};

/// Options for the aig lowering pipeline.
struct CombLoweringPipelineOptions
    : public mlir::PassPipelineOptions<CombLoweringPipelineOptions> {
  PassOptions::Option<bool> disableDatapath{
      *this, "disable-datapath",
      llvm::cl::desc("Disable datapath optimization passes"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> timingAware{
      *this, "timing-aware",
      llvm::cl::desc("Lower operators in a timing-aware fashion"),
      llvm::cl::init(false)};
  PassOptions::Option<TargetIR> targetIR{
      *this, "lowering-target", llvm::cl::desc("Target IR to lower to"),
      llvm::cl::init(TargetIR::AIG)};
  PassOptions::Option<OptimizationStrategy> synthesisStrategy{
      *this, "synthesis-strategy", llvm::cl::desc("Synthesis strategy to use"),
      llvm::cl::values(
          clEnumValN(OptimizationStrategyArea, "area", "Optimize for area"),
          clEnumValN(OptimizationStrategyTiming, "timing",
                     "Optimize for timing")),
      llvm::cl::init(OptimizationStrategyTiming)};
};

/// Options for the synth optimization pipeline.
struct SynthOptimizationPipelineOptions
    : public mlir::PassPipelineOptions<SynthOptimizationPipelineOptions> {
  PassOptions::ListOption<std::string> abcCommands{
      *this, "abc-commands", llvm::cl::desc("ABC passes to run")};

  PassOptions::Option<std::string> abcPath{
      *this, "abc-path", llvm::cl::desc("Path to ABC"), llvm::cl::init("abc")};

  PassOptions::Option<bool> ignoreAbcFailures{
      *this, "ignore-abc-failures",
      llvm::cl::desc("Continue on ABC failure instead of aborting"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> disableWordToBits{
      *this, "disable-word-to-bits",
      llvm::cl::desc("Disable LowerWordToBits pass"), llvm::cl::init(false)};

  PassOptions::Option<bool> disableSOPBalancing{
      *this, "disable-sop-balancing",
      llvm::cl::desc("Disable SOPBalancing pass"), llvm::cl::init(true)};

  PassOptions::Option<bool> timingAware{
      *this, "timing-aware",
      llvm::cl::desc("Lower operators in a timing-aware fashion"),
      llvm::cl::init(false)};

  PassOptions::Option<std::string> cutRewriteDB{
      *this, "cut-rewrite-db",
      llvm::cl::desc("Enable generic cut rewriting using the named database"),
      llvm::cl::init("")};

  PassOptions::Option<unsigned> cutRewriteMaxCutsPerRoot{
      *this, "cut-rewrite-max-cuts-per-root",
      llvm::cl::desc("Maximum number of cuts per root for cut rewriting"),
      llvm::cl::init(4)};

  PassOptions::Option<unsigned> cutRewriteMaxCutInputSize{
      *this, "cut-rewrite-max-cut-input-size",
      llvm::cl::desc("Maximum cut input size for cut rewriting"),
      llvm::cl::init(4)};

  PassOptions::Option<int64_t> cutRewriteConflictLimit{
      *this, "cut-rewrite-conflict-limit",
      llvm::cl::desc("Per-SAT-call conflict budget for SAT-backed cut-rewrite databases"),
      llvm::cl::init(100)};
};

//===----------------------------------------------------------------------===//
// Pipeline Functions
//===----------------------------------------------------------------------===//

/// Populate the synthesis pipelines.
void buildCombLoweringPipeline(mlir::OpPassManager &pm,
                               const CombLoweringPipelineOptions &options);
void buildSynthOptimizationPipeline(
    mlir::OpPassManager &pm, const SynthOptimizationPipelineOptions &options);

/// Register the synthesis pipelines.
void registerSynthesisPipeline();

} // namespace synth
} // namespace circt

#endif // CIRCT_DIALECT_SYNTH_TRANSFORMS_SYNTHESISPIPELINE_H
