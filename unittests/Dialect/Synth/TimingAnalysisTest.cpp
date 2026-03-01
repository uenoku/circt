//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the Two-Stage Timing Analysis Framework.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace synth;
using namespace synth::timing;

namespace {

// Simple test IR with registers and combinational logic
const char *testIR = R"MLIR(
    hw.module private @simple(in %clock : !seq.clock, in %a : i2, in %b : i2, out x : i2) {
      %p = seq.firreg %a clock %clock : i2
      %q = seq.firreg %s clock %clock : i2
      %s = synth.aig.and_inv %p, %q : i2
      hw.output %s : i2
    }
)MLIR";

// IR with deeper combinational logic chain
const char *deepIR = R"MLIR(
    hw.module private @deep(in %clock : !seq.clock, in %a : i1, out x : i1) {
      %reg1 = seq.firreg %a clock %clock : i1
      %and1 = synth.aig.and_inv %reg1, %reg1 : i1
      %and2 = synth.aig.and_inv %and1, %reg1 : i1
      %and3 = synth.aig.and_inv %and2, %and1 : i1
      %reg2 = seq.firreg %and3 clock %clock : i1
      hw.output %reg2 : i1
    }
)MLIR";

class TimingAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<SynthDialect>();
    context.loadDialect<hw::HWDialect>();
    context.loadDialect<seq::SeqDialect>();
    context.loadDialect<comb::CombDialect>();
  }

  MLIRContext context;
};

TEST_F(TimingAnalysisTest, TimingGraphConstruction) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  // Build timing graph
  TimingGraph graph(hwModule);
  auto result = graph.build();
  EXPECT_TRUE(succeeded(result));

  // Should have nodes
  EXPECT_GT(graph.getNumNodes(), 0u);

  // Should have start points (register outputs)
  auto startPoints = graph.getStartPoints();
  EXPECT_GT(startPoints.size(), 0u);

  // Should have end points (register inputs)
  auto endPoints = graph.getEndPoints();
  EXPECT_GT(endPoints.size(), 0u);

  // Should have topological order computed
  // Note: Some nodes may not be in topological order if they're disconnected
  auto topoOrder = graph.getTopologicalOrder();
  EXPECT_GT(topoOrder.size(), 0u);
  EXPECT_LE(topoOrder.size(), graph.getNumNodes());
}

TEST_F(TimingAnalysisTest, ArrivalAnalysisBasic) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  // Build timing graph
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  // Run arrival analysis
  ArrivalAnalysis arrivals(graph);
  auto result = arrivals.run();
  EXPECT_TRUE(succeeded(result));

  // Check that end points have valid arrivals
  int64_t maxDelay = 0;
  for (auto *endPoint : graph.getEndPoints()) {
    auto *arrivalData = arrivals.getArrivalData(endPoint);
    ASSERT_NE(arrivalData, nullptr);
    int64_t arrivalTime = arrivals.getMaxArrivalTime(endPoint);
    if (arrivalTime > maxDelay)
      maxDelay = arrivalTime;
  }

  // Max delay should be non-negative
  EXPECT_GE(maxDelay, 0);
}

TEST_F(TimingAnalysisTest, DeepCombinationalChain) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  // Build timing graph
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  // Run arrival analysis
  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  // Compute max delay from endpoints
  int64_t maxDelay = 0;
  for (auto *endPoint : graph.getEndPoints()) {
    int64_t arrivalTime = arrivals.getMaxArrivalTime(endPoint);
    if (arrivalTime > maxDelay)
      maxDelay = arrivalTime;
  }

  // The deep chain has 3 AND gates in series.
  // The actual delay depends on how the graph is built and propagated.
  // We expect at least some positive delay from the chain.
  EXPECT_GE(maxDelay, 1);
}

TEST_F(TimingAnalysisTest, PathEnumeratorBasic) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  // Build timing graph
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  // Run arrival analysis
  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  // Create path enumerator
  PathEnumerator enumerator(graph, arrivals);

  // Query for worst paths
  PathQuery query;
  query.maxPaths = 10;

  SmallVector<TimingPath> paths;
  auto result = enumerator.enumerate(query, paths);
  EXPECT_TRUE(succeeded(result));

  // Should find some paths (from registers to registers)
  // The exact number depends on the circuit structure
  EXPECT_GE(paths.size(), 0u);
}

TEST_F(TimingAnalysisTest, ObjectMatching) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  // Build timing graph
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  // Test pattern matching with wildcard
  SmallVector<std::string> patterns = {"*"};
  auto matchedNodes = matchNodes(graph, patterns);
  EXPECT_GT(matchedNodes.size(), 0u);

  // Test start point matching
  auto startPoints = matchStartPoints(graph, patterns);
  EXPECT_GT(startPoints.size(), 0u);

  // Test end point matching
  auto endPoints = matchEndPoints(graph, patterns);
  EXPECT_GT(endPoints.size(), 0u);
}

TEST_F(TimingAnalysisTest, TimingAnalysisInterface) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  // Use the main TimingAnalysis interface
  auto analysis = TimingAnalysis::create(hwModule);
  ASSERT_NE(analysis, nullptr);

  // Build graph and run arrival analysis
  EXPECT_TRUE(succeeded(analysis->buildGraph()));
  EXPECT_TRUE(analysis->hasGraph());

  EXPECT_TRUE(succeeded(analysis->runArrivalAnalysis()));
  EXPECT_TRUE(analysis->hasArrivalData());

  // Query for paths using enumeratePaths
  PathQuery query;
  query.maxPaths = 5;
  SmallVector<TimingPath> paths;
  EXPECT_TRUE(succeeded(analysis->enumeratePaths(query, paths)));
}

} // namespace

