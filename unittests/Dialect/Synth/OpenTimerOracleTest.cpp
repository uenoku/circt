//===- OpenTimerOracleTest.cpp - OpenTimer backend microbenchmark ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hand-checkable microbenchmark for the OpenTimer backend: one AND2
// feeding one flop against an extracted ASAP7 subset (see
// test/Dialect/Synth/Inputs/tiny-timing.lib). Expected values are
// cross-checked against OpenSTA with matching boundary conditions
// (10ps input slew, 1ps clock slew, 0.5ns period):
//
//   arrival(D)  = 15.365ps (OpenSTA) vs 15.36ps (OpenTimer)
//   required(D) = 485.666ps (both, exact)
//   slack(D)    = 470.301ps
//
// The flop output is unconstrained (no RAT), so exactly one endpoint
// (the flop D pin, both transitions) is reported.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingDesignBuilder.h"
#include "circt/Dialect/Synth/Timing/TimingOracle.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth::timing;

namespace {

const char *kTinyLibPath = CIRCT_TINY_LIB_PATH;

const char *andFlopIR = R"MLIR(
    hw.module.extern @AND2x4_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1)
    hw.module.extern @DFFHQNx1_ASAP7_75t_R(in %D : i1, in %CLK : i1, out QN : i1)
    hw.module @top(in %a : i1, in %clk : i1, out qn : i1) {
      %y = hw.instance "g1" @AND2x4_ASAP7_75t_R(A: %a: i1, B: %a: i1) -> (Y: i1)
      %q = hw.instance "f1" @DFFHQNx1_ASAP7_75t_R(D: %y: i1, CLK: %clk: i1) -> (QN: i1)
      hw.output %q : i1
    }
    )MLIR";

TEST(OpenTimerOracle, AndIntoFlop) {
  MLIRContext ctx;
  ctx.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  synth::SynthDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(andFlopIR, &ctx);
  ASSERT_TRUE(module);

  auto top = dyn_cast<hw::HWModuleOp>(module->lookupSymbol("top"));
  ASSERT_TRUE(top);
  DenseSet<StringAttr> lib{StringAttr::get(&ctx, "AND2x4_ASAP7_75t_R"),
                           StringAttr::get(&ctx, "DFFHQNx1_ASAP7_75t_R")};
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto design = builder.build(*module);
  for (const auto &diag : builder.getDiagnostics().unsupported)
    llvm::errs() << "builder diag: " << diag.message.getValue() << "\n";
  ASSERT_TRUE(succeeded(design));

  // Microbenchmark Liberty model (extracted ASAP7 subset).

  TimingRequest request;
  request.design = &*design;
  request.libertyFiles = {kTinyLibPath};
  TimingClock clock;
  clock.name = StringAttr::get(&ctx, "clk");
  clock.source.kind = TimingObjectKind::TopPort;
  clock.source.portName = StringAttr::get(&ctx, "clk");
  clock.source.portIndex = 1;
  clock.source.bit = 0;
  clock.period = TimePs{500};
  request.constraints.clocks.push_back(clock);
  TimingPortConstraint inputA;
  inputA.port.kind = TimingObjectKind::TopPort;
  inputA.port.portName = StringAttr::get(&ctx, "a");
  inputA.port.portIndex = 0;
  inputA.port.bit = 0;
  inputA.lateArrival = TimePs{0};
  inputA.riseSlew = TimePs{10};
  inputA.fallSlew = TimePs{10};
  request.constraints.ports.push_back(inputA);
  // No output RAT: the flop Q output stays unconstrained.

  auto oracle = createOpenTimerTimingOracle(&ctx);
  ASSERT_TRUE(oracle);
  EXPECT_EQ(oracle->getName(), "opentimer");
  EXPECT_TRUE(oracle->getCapabilities().setupMax);
  auto result = oracle->analyze(request);
  ASSERT_TRUE(succeeded(result));

  // Exactly one endpoint (flop D); Q is unconstrained.
  ASSERT_EQ(result->endpoints.size(), 1u);
  const auto &endpoint = result->endpoints.front();
  EXPECT_EQ(endpoint.endpoint.portName, StringAttr::get(&ctx, "D"));
  ASSERT_EQ(endpoint.transitions.size(), 2u);
  EXPECT_EQ(endpoint.transitions[0].transition, TimingTransition::Rise);
  EXPECT_EQ(endpoint.transitions[1].transition, TimingTransition::Fall);
  const auto &tr = endpoint.transitions.front();
  ASSERT_TRUE(tr.arrival.has_value());
  ASSERT_TRUE(tr.required.has_value());
  EXPECT_NEAR(tr.arrival->value, 15, 1);
  EXPECT_NEAR(tr.required->value, 486, 1);
  EXPECT_NEAR(tr.slack.value, 470, 1);
  // Launch tracing needs violated paths; met endpoints omit it.
  EXPECT_FALSE(tr.worstLaunch.has_value());
  for (const auto &t : endpoint.transitions) {
    EXPECT_TRUE(t.arrival.has_value());
    EXPECT_TRUE(t.required.has_value());
  }

  EXPECT_TRUE(result->wns.has_value());
  EXPECT_NEAR(result->wns->value, 470, 1);
  EXPECT_EQ(result->tns->value, 0);
  EXPECT_EQ(result->failingEndpoints, 0u);
  EXPECT_EQ(result->coverage.analyzedEndpoints, 1u);
  EXPECT_EQ(result->coverage.unconstrainedEndpoints, 0u);
}

TEST(OpenTimerOracle, HoldMinAnalysis) {
  MLIRContext ctx;
  ctx.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  synth::SynthDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(andFlopIR, &ctx);
  ASSERT_TRUE(module);
  DenseSet<StringAttr> lib{StringAttr::get(&ctx, "AND2x4_ASAP7_75t_R"),
                           StringAttr::get(&ctx, "DFFHQNx1_ASAP7_75t_R")};
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto design = builder.build(*module);
  ASSERT_TRUE(succeeded(design));

  TimingRequest request;
  request.design = &*design;
  request.libertyFiles = {kTinyLibPath};
  request.constraints.analysis = TimingAnalysisKind::HoldMin;
  TimingClock clock;
  clock.name = StringAttr::get(&ctx, "clk");
  clock.source.kind = TimingObjectKind::TopPort;
  clock.source.portName = StringAttr::get(&ctx, "clk");
  clock.source.portIndex = 1;
  clock.source.bit = 0;
  clock.period = TimePs{500};
  request.constraints.clocks.push_back(clock);
  TimingPortConstraint inputA;
  inputA.port.kind = TimingObjectKind::TopPort;
  inputA.port.portName = StringAttr::get(&ctx, "a");
  inputA.port.portIndex = 0;
  inputA.port.bit = 0;
  inputA.lateArrival = TimePs{0};
  inputA.riseSlew = TimePs{10};
  inputA.fallSlew = TimePs{10};
  request.constraints.ports.push_back(inputA);

  auto oracle = createOpenTimerTimingOracle(&ctx);
  auto result = oracle->analyze(request);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->analysis, TimingAnalysisKind::HoldMin);
  // Hold checks with ideal clocks are met (positive slack); recovery and
  // min-pulse checks are excluded as unsupported, not mixed in.
  ASSERT_EQ(result->endpoints.size(), 1u);
  for (const auto &tr : result->endpoints.front().transitions)
    EXPECT_GE(tr.slack.value, 0);
  EXPECT_EQ(result->failingEndpoints, 0u);
}

TEST(OpenTimerOracle, BufferedClockPropagates) {
  // Two buffers on the clock network: capture required shifts later by
  // the insertion delay (input-to-flop gains slack). Validates propagated
  // (non-ideal) clocks without extra code (network in graph suffices).
  const char *ir = R"MLIR(
    hw.module.extern @BUFx2_ASAP7_75t_R(in %A : i1, out Y : i1)
    hw.module.extern @AND2x4_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1)
    hw.module.extern @DFFHQNx1_ASAP7_75t_R(in %D : i1, in %CLK : i1, out QN : i1)
    hw.module @top(in %a : i1, in %clk : i1, out qn : i1) {
      %cb1 = hw.instance "cb1" @BUFx2_ASAP7_75t_R(A: %clk: i1) -> (Y: i1)
      %cb2 = hw.instance "cb2" @BUFx2_ASAP7_75t_R(A: %cb1: i1) -> (Y: i1)
      %y = hw.instance "g1" @AND2x4_ASAP7_75t_R(A: %a: i1, B: %a: i1) -> (Y: i1)
      %q = hw.instance "f1" @DFFHQNx1_ASAP7_75t_R(D: %y: i1, CLK: %cb2: i1) -> (QN: i1)
      hw.output %q : i1
    }
    )MLIR";
  MLIRContext ctx;
  ctx.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                  synth::SynthDialect>();
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  ASSERT_TRUE(module);
  DenseSet<StringAttr> lib{StringAttr::get(&ctx, "BUFx2_ASAP7_75t_R"),
                           StringAttr::get(&ctx, "AND2x4_ASAP7_75t_R"),
                           StringAttr::get(&ctx, "DFFHQNx1_ASAP7_75t_R")};
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto design = builder.build(*module);
  ASSERT_TRUE(succeeded(design));

  TimingRequest request;
  request.design = &*design;
  request.libertyFiles = {kTinyLibPath};
  TimingClock clock;
  clock.name = StringAttr::get(&ctx, "clk");
  clock.source.kind = TimingObjectKind::TopPort;
  clock.source.portName = StringAttr::get(&ctx, "clk");
  clock.source.portIndex = 1;
  clock.source.bit = 0;
  clock.period = TimePs{500};
  request.constraints.clocks.push_back(clock);
  TimingPortConstraint inputA;
  inputA.port.kind = TimingObjectKind::TopPort;
  inputA.port.portName = StringAttr::get(&ctx, "a");
  inputA.port.portIndex = 0;
  inputA.port.bit = 0;
  inputA.lateArrival = TimePs{0};
  inputA.riseSlew = TimePs{10};
  inputA.fallSlew = TimePs{10};
  request.constraints.ports.push_back(inputA);

  auto oracle = createOpenTimerTimingOracle(&ctx);
  auto result = oracle->analyze(request);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->endpoints.size(), 1u);
  const auto &tr = result->endpoints.front().transitions.front();
  ASSERT_TRUE(tr.arrival.has_value());
  ASSERT_TRUE(tr.required.has_value());
  // Data arrival unchanged (15ps); required pushed later by clock insertion
  // (STA ideal: 486ps; propagated must exceed it).
  EXPECT_NEAR(tr.arrival->value, 15, 1);
  EXPECT_GT(tr.required->value, 486);
}

} // namespace
