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
#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Synth/Analysis/Timing/RequiredTimeAnalysis.h"
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

  TimingGraph graph(hwModule);
  auto result = graph.build();
  EXPECT_TRUE(succeeded(result));

  EXPECT_GT(graph.getNumNodes(), 0u);
  EXPECT_GT(graph.getStartPoints().size(), 0u);
  EXPECT_GT(graph.getEndPoints().size(), 0u);
  EXPECT_GT(graph.getTopologicalOrder().size(), 0u);
  EXPECT_LE(graph.getTopologicalOrder().size(), graph.getNumNodes());
}

TEST_F(TimingAnalysisTest, TopologicalOrderCorrectness) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  // For every arc (u -> v), u must appear before v in topo order
  auto topoOrder = graph.getTopologicalOrder();
  llvm::DenseMap<TimingNode *, size_t> position;
  for (size_t i = 0; i < topoOrder.size(); ++i)
    position[topoOrder[i]] = i;

  for (auto *node : topoOrder) {
    for (auto *arc : node->getFanout()) {
      auto *succ = arc->getTo();
      auto fromIt = position.find(node);
      auto toIt = position.find(succ);
      if (fromIt != position.end() && toIt != position.end())
        EXPECT_LT(fromIt->second, toIt->second)
            << "Arc from " << node->getName().str() << " to "
            << succ->getName().str() << " violates topo order";
    }
  }
}

TEST_F(TimingAnalysisTest, ArrivalAnalysisBasic) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  auto result = arrivals.run();
  EXPECT_TRUE(succeeded(result));

  int64_t maxDelay = 0;
  for (auto *endPoint : graph.getEndPoints()) {
    auto *arrivalData = arrivals.getArrivalData(endPoint);
    ASSERT_NE(arrivalData, nullptr);
    int64_t arrivalTime = arrivals.getMaxArrivalTime(endPoint);
    if (arrivalTime > maxDelay)
      maxDelay = arrivalTime;
  }
  EXPECT_GE(maxDelay, 0);
}

TEST_F(TimingAnalysisTest, DeepCombinationalChain) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  int64_t maxDelay = 0;
  for (auto *endPoint : graph.getEndPoints()) {
    int64_t arrivalTime = arrivals.getMaxArrivalTime(endPoint);
    if (arrivalTime > maxDelay)
      maxDelay = arrivalTime;
  }
  EXPECT_GE(maxDelay, 1);
}

TEST_F(TimingAnalysisTest, DelayModelPluggable) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  // Build with unit delay model
  UnitDelayModel unitModel;
  TimingGraph graph1(hwModule);
  ASSERT_TRUE(succeeded(graph1.build(&unitModel)));
  EXPECT_EQ(graph1.getDelayModelName(), "unit");

  ArrivalAnalysis arrivals1(graph1);
  ASSERT_TRUE(succeeded(arrivals1.run()));

  // Build with AIG level delay model (default)
  AIGLevelDelayModel aigModel;
  TimingGraph graph2(hwModule);
  ASSERT_TRUE(succeeded(graph2.build(&aigModel)));
  EXPECT_EQ(graph2.getDelayModelName(), "aig-level");

  ArrivalAnalysis arrivals2(graph2);
  ASSERT_TRUE(succeeded(arrivals2.run()));

  // Both should produce valid results; exact values may differ
  int64_t maxUnit = 0, maxAIG = 0;
  for (auto *ep : graph1.getEndPoints())
    maxUnit = std::max(maxUnit, arrivals1.getMaxArrivalTime(ep));
  for (auto *ep : graph2.getEndPoints())
    maxAIG = std::max(maxAIG, arrivals2.getMaxArrivalTime(ep));

  EXPECT_GE(maxUnit, 1);
  EXPECT_GE(maxAIG, 1);
}

TEST_F(TimingAnalysisTest, RequiredTimeAnalysisTest) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  // Unconstrained: slack should be 0 at critical endpoint
  RequiredTimeAnalysis rat(graph, arrivals);
  ASSERT_TRUE(succeeded(rat.run()));

  // Worst slack should be 0 when unconstrained
  EXPECT_EQ(rat.getWorstSlack(), 0);

  // All endpoint slacks should be >= 0 when unconstrained
  for (auto *ep : graph.getEndPoints())
    EXPECT_GE(rat.getSlack(ep), 0);
}

TEST_F(TimingAnalysisTest, SlackNegative) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  // Find critical path delay
  int64_t maxDelay = 0;
  for (auto *ep : graph.getEndPoints())
    maxDelay = std::max(maxDelay, arrivals.getMaxArrivalTime(ep));

  // Set clock period smaller than critical path -> negative slack
  if (maxDelay > 1) {
    RequiredTimeAnalysis::Options ratOpts;
    ratOpts.clockPeriod = 1; // Very tight constraint
    RequiredTimeAnalysis rat(graph, arrivals, ratOpts);
    ASSERT_TRUE(succeeded(rat.run()));
    EXPECT_LT(rat.getWorstSlack(), 0);
  }
}

TEST_F(TimingAnalysisTest, PathEnumeratorBasic) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  PathEnumerator enumerator(graph, arrivals);

  PathQuery query;
  query.maxPaths = 10;

  SmallVector<TimingPath> paths;
  auto result = enumerator.enumerate(query, paths);
  EXPECT_TRUE(succeeded(result));
  EXPECT_GE(paths.size(), 0u);
}

TEST_F(TimingAnalysisTest, KWorstPathsOrdering) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  ArrivalAnalysis arrivals(graph);
  ASSERT_TRUE(succeeded(arrivals.run()));

  PathEnumerator enumerator(graph, arrivals);

  SmallVector<TimingPath> paths;
  ASSERT_TRUE(succeeded(enumerator.getKWorstPaths(10, paths)));

  // Verify descending order
  for (size_t i = 1; i < paths.size(); ++i)
    EXPECT_GE(paths[i - 1].getDelay(), paths[i].getDelay());

  // K=1 should give critical path
  if (!paths.empty()) {
    SmallVector<TimingPath> one;
    ASSERT_TRUE(succeeded(enumerator.getKWorstPaths(1, one)));
    ASSERT_EQ(one.size(), 1u);
    EXPECT_EQ(one[0].getDelay(), paths[0].getDelay());
  }
}

TEST_F(TimingAnalysisTest, ObjectMatching) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build()));

  SmallVector<std::string> patterns = {"*"};
  auto matchedNodes = matchNodes(graph, patterns);
  EXPECT_GT(matchedNodes.size(), 0u);

  auto startPoints = matchStartPoints(graph, patterns);
  EXPECT_GT(startPoints.size(), 0u);

  auto endPoints = matchEndPoints(graph, patterns);
  EXPECT_GT(endPoints.size(), 0u);
}

TEST_F(TimingAnalysisTest, ReportTimingSmoke) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(deepIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("deep");
  ASSERT_TRUE(hwModule);

  auto analysis = TimingAnalysis::create(hwModule);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));

  // Report to a string
  std::string report;
  llvm::raw_string_ostream os(report);
  analysis->reportTiming(os, 5);

  // Check report contains expected sections
  EXPECT_NE(report.find("=== Timing Report ==="), std::string::npos);
  EXPECT_NE(report.find("Module: deep"), std::string::npos);
  EXPECT_NE(report.find("Delay Model:"), std::string::npos);
  EXPECT_NE(report.find("Worst Slack:"), std::string::npos);
  EXPECT_NE(report.find("Critical Paths"), std::string::npos);
}

TEST_F(TimingAnalysisTest, FullPipeline) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  auto analysis = TimingAnalysis::create(hwModule);
  ASSERT_NE(analysis, nullptr);

  // Full analysis
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));
  EXPECT_TRUE(analysis->hasGraph());
  EXPECT_TRUE(analysis->hasArrivalData());
  EXPECT_TRUE(analysis->hasRequiredTimeData());

  // Get worst slack
  int64_t worstSlack = analysis->getWorstSlack();
  EXPECT_EQ(worstSlack, 0); // Unconstrained

  // Report timing
  std::string report;
  llvm::raw_string_ostream os(report);
  analysis->reportTiming(os, 5);
  EXPECT_FALSE(report.empty());

  // Get paths
  SmallVector<TimingPath> paths;
  EXPECT_TRUE(succeeded(analysis->getKWorstPaths(5, paths)));
}

TEST_F(TimingAnalysisTest, TimingAnalysisInterface) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  auto analysis = TimingAnalysis::create(hwModule);
  ASSERT_NE(analysis, nullptr);

  EXPECT_TRUE(succeeded(analysis->buildGraph()));
  EXPECT_TRUE(analysis->hasGraph());

  EXPECT_TRUE(succeeded(analysis->runArrivalAnalysis()));
  EXPECT_TRUE(analysis->hasArrivalData());

  PathQuery query;
  query.maxPaths = 5;
  SmallVector<TimingPath> paths;
  EXPECT_TRUE(succeeded(analysis->enumeratePaths(query, paths)));
}

} // namespace
