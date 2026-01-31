//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements trivial mockturtle passes that are simple bindings
// to mockturtle transformation algorithms.
//
//===----------------------------------------------------------------------===//

#include "NetworkConversion.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "synth-mockturtle-passes"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_MOCKTURTLEREFACTOR
#define GEN_PASS_DEF_MOCKTURTLEFUNCTIONALREDUCTION
#define GEN_PASS_DEF_MOCKTURTLEMIGALGEBRAICREWRITEDEPTH
#define GEN_PASS_DEF_MOCKTURTLESOPBALANCING
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;
using namespace circt::synth::mockturtle_integration;

namespace {

//===----------------------------------------------------------------------===//
// Refactor Pass
//===----------------------------------------------------------------------===//

/// Refactor pass using mockturtle algorithms for logic optimization.
/// This pass performs logic refactoring using reconvergence-driven cuts
/// to optimize circuit structure.
struct MockturtleRefactorPass
    : public impl::MockturtleRefactorBase<MockturtleRefactorPass> {
  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Running Mockturtle Refactor pass on module: "
                            << module.getModuleName() << "\n");
    if (failed(runNetworkTransforms(module.getBodyBlock(), runSOPRefactoring)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// Functional Reduction Pass
//===----------------------------------------------------------------------===//

/// Functional reduction pass using mockturtle's SAT-based optimization.
/// This pass identifies and removes functionally redundant nodes using
/// SAT-based equivalence checking with simulation-based filtering.
struct MockturtleFunctionalReductionPass
    : public impl::MockturtleFunctionalReductionBase<
          MockturtleFunctionalReductionPass> {
  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "Running Mockturtle Functional Reduction pass on module: "
               << module.getModuleName() << "\n");
    if (failed(runNetworkTransforms(module.getBodyBlock(),
                                    runFunctionalReduction)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// MIG Algebraic Rewrite Depth Pass
//===----------------------------------------------------------------------===//

/// MIG algebraic depth rewriting pass for depth optimization.
/// This pass applies MIG algebraic rewriting rules to reduce circuit depth
/// using DFS strategy.
struct MockturtleMIGAlgebraicRewriteDepthPass
    : public impl::MockturtleMIGAlgebraicRewriteDepthBase<
          MockturtleMIGAlgebraicRewriteDepthPass> {
  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "Running Mockturtle MIG Algebraic Rewrite Depth pass on "
                  "module: "
               << module.getModuleName() << "\n");
    if (failed(runMIGNetworkTransforms(module.getBodyBlock(),
                                       runMIGAlgebraicRewriteDepth)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// SOP Balancing Pass
//===----------------------------------------------------------------------===//
/// SOP balancing pass using mockturtle's LUT-based balancing algorithm.
/// This pass restructures the circuit using SOP balancing to optimize
/// depth and area.
struct MockturtleSOPBalancingPass
    : public impl::MockturtleSOPBalancingBase<MockturtleSOPBalancingPass> {
  void runOnOperation() override {
    auto module = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "Running Mockturtle SOP Balancing pass on module: "
               << module.getModuleName() << "\n");
    if (failed(runNetworkTransforms(module.getBodyBlock(), runSOPBalancing)))
      signalPassFailure();
  }
};

} // namespace
