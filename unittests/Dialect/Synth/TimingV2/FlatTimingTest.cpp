//===- FlatTimingTest.cpp - Flat timing framework tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Analysis/TimingV2/FlatTiming.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth;
using namespace circt::synth::timingv2;

namespace {

class FlatTimingTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.getOrLoadDialect<hw::HWDialect>();
    context.getOrLoadDialect<comb::CombDialect>();
    context.getOrLoadDialect<datapath::DatapathDialect>();
    context.getOrLoadDialect<seq::SeqDialect>();
    context.getOrLoadDialect<synth::SynthDialect>();
    context.allowUnregisteredDialects();
  }

  OwningOpRef<ModuleOp> parse(StringRef text) {
    return parseSourceString<ModuleOp>(text, &context);
  }

  hw::HWModuleOp getModule(ModuleOp module, StringRef name) {
    for (auto hwModule : module.getOps<hw::HWModuleOp>())
      if (hwModule.getModuleName() == name)
        return hwModule;
    return {};
  }

  MLIRContext context;
};

class ObjectiveDelayModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    if (!ctx.op)
      return {0, 0.0};
    return {ctx.objective == TimingDelayObjective::Min ? 1 : 5, 0.0};
  }

  StringRef getName() const override { return "objective-test"; }
};

static TimingSemanticPoint semanticPoint(
    Value value, uint32_t bit,
    TimingPointKind kind = TimingPointKind::ValueBit,
    StringRef name = {}, Operation *owner = nullptr) {
  TimingSemanticPoint point;
  point.value = value;
  point.bit = bit;
  point.kind = kind;
  point.name = name.str();
  point.owner = owner;
  return point;
}

class OverrideAndSemanticsProvider final : public TimingSemanticsProvider {
public:
  FailureOr<TimingSemantics> describe(Operation *op) const override {
    if (!isa<comb::AndOp>(op))
      return fallback.describe(op);

    TimingSemantics semantics;
    semantics.op = op;
    auto result = op->getResult(0);
    auto to = semanticPoint(result, 0, TimingPointKind::ValueBit, {}, op);
    for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
      TimingSemanticArc arc;
      arc.from = semanticPoint(operand, 0);
      arc.to = to;
      arc.kind = TimingArcKind::Synthetic;
      arc.fixedDelay = 9;
      arc.op = op;
      arc.inputIndex = static_cast<int32_t>(operandIndex);
      arc.outputIndex = 0;
      arc.token = "policy_and";
      semantics.arcs.push_back(std::move(arc));
    }
    return semantics;
  }

private:
  DefaultTimingSemanticsProvider fallback;
};

const char *datamoveIR = R"MLIR(
module {
  hw.module @datamove(in %a : i2, in %b : i2, out y : i4) {
    %c = comb.concat %a, %b : i2, i2
    hw.output %c : i4
  }
}
)MLIR";

const char *partialProductIR = R"MLIR(
module {
  hw.module @partial_product(in %a : i2, in %b : i2, out y : i2) {
    %pp:2 = datapath.partial_product %a, %b : (i2, i2) -> (i2, i2)
    %out = comb.xor %pp#0, %pp#1 : i2
    hw.output %out : i2
  }
}
)MLIR";

const char *registerIR = R"MLIR(
module {
  hw.module @regs(in %clock : !seq.clock, in %a : i2, out y : i2) {
    %r = seq.firreg %a clock %clock {"name" = "r"} : i2
    hw.output %r : i2
  }
}
)MLIR";

const char *propagationIR = R"MLIR(
module {
  hw.module @prop(in %a : i1, in %b : i1, out y : i1) {
    %fast = comb.and %a, %a : i1
    %slow1 = comb.and %b, %b : i1
    %slow2 = comb.and %slow1, %b : i1
    %out = comb.xor %fast, %slow2 : i1
    hw.output %out : i1
  }
}
)MLIR";

const char *datapathDynamicIR = R"MLIR(
module {
  hw.module @dynamic(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
    %slow = comb.and %a, %a : i1
    %comp:2 = datapath.compress %slow, %b, %c : i1 [3 -> 2]
    %out = comb.xor %comp#0, %comp#1 : i1
    hw.output %out : i1
  }
}
)MLIR";

const char *datapathWideCompressIR = R"MLIR(
module {
  hw.module @wide_compress(in %a : i2, in %b : i2, in %c : i2, out y : i2) {
    %comp:2 = datapath.compress %a, %b, %c : i2 [3 -> 2]
    %out = comb.xor %comp#0, %comp#1 : i2
    hw.output %out : i2
  }
}
)MLIR";

const char *datapathFourToTwoCompressIR = R"MLIR(
module {
  hw.module @compress42(in %a : i2, in %b : i2, in %c : i2, in %d : i2, out y : i2) {
    %comp:2 = datapath.compress %a, %b, %c, %d : i2 [4 -> 2]
    %out = comb.xor %comp#0, %comp#1 : i2
    hw.output %out : i2
  }
}
)MLIR";

const char *unknownOpIR = R"MLIR(
module {
  hw.module @unknown(in %a : i2, in %b : i2, out y : i2) {
    %u = "test.unknown_comb"(%a, %b) : (i2, i2) -> i2
    hw.output %u : i2
  }
}
)MLIR";

TEST_F(FlatTimingTest, BuildsPreciseConcatBitArcs) {
  auto module = parse(datamoveIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "datamove");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  auto concat = *hwModule.getOps<comb::ConcatOp>().begin();
  auto a0 = network.findValueBit(hwModule.getBodyBlock()->getArgument(0), 0);
  auto b0 = network.findValueBit(hwModule.getBodyBlock()->getArgument(1), 0);
  auto c0 = network.findValueBit(concat.getResult(), 0);
  auto c2 = network.findValueBit(concat.getResult(), 2);
  ASSERT_TRUE(a0.isValid());
  ASSERT_TRUE(b0.isValid());
  ASSERT_TRUE(c0.isValid());
  ASSERT_TRUE(c2.isValid());

  bool sawBToLow = false;
  for (auto arcIndex : network.getPoint(c0)->fanin) {
    auto *arc = network.getArc(arcIndex);
    sawBToLow |= arc->from == b0 && arc->token == "concat";
  }
  EXPECT_TRUE(sawBToLow);

  bool sawAToHigh = false;
  for (auto arcIndex : network.getPoint(c2)->fanin) {
    auto *arc = network.getArc(arcIndex);
    sawAToHigh |= arc->from == a0 && arc->token == "concat";
  }
  EXPECT_TRUE(sawAToHigh);
}

TEST_F(FlatTimingTest, BuildsRegisterCutPoints) {
  auto module = parse(registerIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "regs");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  auto reg = *hwModule.getOps<seq::FirRegOp>().begin();
  auto q0 = network.findValueBit(reg.getResult(), 0);
  ASSERT_TRUE(q0.isValid());
  EXPECT_EQ(network.getPoint(q0)->kind, TimingPointKind::CutStart);

  bool sawCutEnd = false;
  for (auto id : network.getEndPoints()) {
    auto *point = network.getPoint(id);
    sawCutEnd |= point->kind == TimingPointKind::CutEnd;
  }
  EXPECT_TRUE(sawCutEnd);
}

TEST_F(FlatTimingTest, BuildsFromNonModuleRootOperation) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  auto xorOp = *hwModule.getOps<comb::XorOp>().begin();
  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(xorOp.getOperation())));

  EXPECT_EQ(network.getStartPoints().size(), 2u);
  EXPECT_EQ(network.getEndPoints().size(), 1u);

  auto result = TimingPropagator::run(network);
  ASSERT_TRUE(succeeded(result));
  auto *endState = result->getState(network.getEndPoints().front());
  ASSERT_TRUE(endState->hasArrival);
  EXPECT_EQ(endState->arrival, 1);
}

TEST_F(FlatTimingTest, PropagatesSetupMaxAndHoldMinOnSameGraph) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));
  ASSERT_EQ(network.getEndPoints().size(), 1u);

  TimingPropagationOptions setupOptions;
  setupOptions.objective = TimingObjective::SetupMax;
  auto setup = TimingPropagator::run(network, setupOptions);
  ASSERT_TRUE(succeeded(setup));
  auto *setupEnd = setup->getState(network.getEndPoints().front());
  ASSERT_TRUE(setupEnd->hasArrival);
  EXPECT_EQ(setupEnd->arrival, 3);

  TimingPropagationOptions holdOptions;
  holdOptions.objective = TimingObjective::HoldMin;
  auto hold = TimingPropagator::run(network, holdOptions);
  ASSERT_TRUE(succeeded(hold));
  auto *holdEnd = hold->getState(network.getEndPoints().front());
  ASSERT_TRUE(holdEnd->hasArrival);
  EXPECT_EQ(holdEnd->arrival, 2);
}

TEST_F(FlatTimingTest, StoresSeparateSetupAndHoldArcDelays) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  ObjectiveDelayModel delayModel;
  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule, &delayModel)));

  auto setup = TimingPropagator::run(network);
  ASSERT_TRUE(succeeded(setup));
  EXPECT_EQ(setup->getState(network.getEndPoints().front())->arrival, 15);

  TimingPropagationOptions holdOptions;
  holdOptions.objective = TimingObjective::HoldMin;
  auto hold = TimingPropagator::run(network, holdOptions);
  ASSERT_TRUE(succeeded(hold));
  EXPECT_EQ(hold->getState(network.getEndPoints().front())->arrival, 2);
}

TEST_F(FlatTimingTest, SemanticsProviderOverridesDefaultOperationSemantics) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  OverrideAndSemanticsProvider provider;
  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule, /*delayModel=*/nullptr,
                                      &provider)));

  auto result = TimingPropagator::run(network);
  ASSERT_TRUE(succeeded(result));
  auto *endState = result->getState(network.getEndPoints().front());
  ASSERT_TRUE(endState->hasArrival);
  EXPECT_EQ(endState->arrival, 19);

  TimingPathReconstructor reconstructor(network, *result);
  auto path = reconstructor.reconstructTo(network.getEndPoints().front());
  ASSERT_TRUE(path);
  bool sawPolicyArc = false;
  for (auto step : path->steps) {
    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    sawPolicyArc |= arc && arc->token == "policy_and";
  }
  EXPECT_TRUE(sawPolicyArc);
}

TEST_F(FlatTimingTest, ReconstructsPathAfterPropagation) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));
  auto result = TimingPropagator::run(network);
  ASSERT_TRUE(succeeded(result));

  TimingPathReconstructor reconstructor(network, *result);
  auto path = reconstructor.reconstructTo(network.getEndPoints().front());
  ASSERT_TRUE(path);
  EXPECT_EQ(path->delay, 3);
  ASSERT_GE(path->steps.size(), 4u);

  bool sawLogicArc = false;
  for (auto step : path->steps) {
    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    sawLogicArc |= arc && (arc->token == "and" || arc->token == "xor");
  }
  EXPECT_TRUE(sawLogicArc);
}

TEST_F(FlatTimingTest, ReportsCriticalTimingPath) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  auto path = reconstructCriticalPath(network);
  ASSERT_TRUE(succeeded(path));
  EXPECT_EQ(path->delay, 3);
  ASSERT_GE(path->steps.size(), 4u);

  std::string report;
  llvm::raw_string_ostream os(report);
  ASSERT_TRUE(succeeded(printCriticalTimingReport(network, os)));
  os.flush();

  EXPECT_TRUE(StringRef(report).contains("TimingV2 critical path report"));
  EXPECT_TRUE(StringRef(report).contains("objective: setup_max"));
  EXPECT_TRUE(StringRef(report).contains("delay: 3"));
  EXPECT_TRUE(StringRef(report).contains("via and"));
  EXPECT_TRUE(StringRef(report).contains("via xor"));
  EXPECT_TRUE(StringRef(report).contains("via root_output"));
}

TEST_F(FlatTimingTest, SpeculatesWorstEndpointReplacementArrival) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  auto context = TimingSpeculationContext::create(network);
  ASSERT_TRUE(succeeded(context));
  EXPECT_EQ(context->getWorstEndpointDelay(), 3);

  auto xorOp = *hwModule.getOps<comb::XorOp>().begin();
  auto xorArrival = context->getArrival(xorOp.getResult(), 0);
  ASSERT_TRUE(succeeded(xorArrival));
  EXPECT_EQ(*xorArrival, 3);
  EXPECT_TRUE(context->isOnWorstEndpointPath(xorOp.getResult(), 0));

  TimingArrivalReplacement replacement;
  replacement.value = xorOp.getResult();
  replacement.bit = 0;
  replacement.arrival = 1;
  auto endpointDelay =
      context->speculateEndpointDelay(context->getWorstEndpointPath().end,
                                      replacement);
  ASSERT_TRUE(succeeded(endpointDelay));
  EXPECT_EQ(*endpointDelay, 1);

  auto speculation = context->speculateWorstEndpointDelay(replacement);
  ASSERT_TRUE(succeeded(speculation));
  EXPECT_TRUE(speculation->affectedWorstEndpointPath);
  EXPECT_EQ(speculation->baselineDelay, 3);
  EXPECT_EQ(speculation->oldArrival, 3);
  EXPECT_EQ(speculation->newArrival, 1);
  EXPECT_EQ(speculation->predictedDelay, 1);

  auto fastAnd = *hwModule.getOps<comb::AndOp>().begin();
  EXPECT_FALSE(context->isOnWorstEndpointPath(fastAnd.getResult(), 0));
  replacement.value = fastAnd.getResult();
  replacement.arrival = 0;
  auto unaffected = context->speculateWorstEndpointDelay(replacement);
  ASSERT_TRUE(succeeded(unaffected));
  EXPECT_FALSE(unaffected->affectedWorstEndpointPath);
  EXPECT_EQ(unaffected->predictedDelay, 3);
}

TEST_F(FlatTimingTest, DatapathSemanticsChoosesAreaCompressorWithSlack) {
  auto module = parse(datapathDynamicIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "dynamic");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  TimingPropagationOptions options;
  options.defaultRequiredTime = 10;
  DatapathTimingSemanticsProvider provider;
  ASSERT_TRUE(succeeded(network.refineDynamicSemantics(provider, options)));

  auto path = reconstructCriticalPath(network, options);
  ASSERT_TRUE(succeeded(path));
  EXPECT_EQ(path->delay, 4);

  bool sawAreaArc = false;
  for (auto step : path->steps) {
    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    sawAreaArc |= arc && arc->token == "compressor_greedy_area";
  }
  EXPECT_TRUE(sawAreaArc);
}

TEST_F(FlatTimingTest, RequiredTimeConstraintsSelectFastDatapathSemantics) {
  auto module = parse(datapathDynamicIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "dynamic");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  Operation *compress = nullptr;
  hwModule.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "datapath.compress")
      compress = op;
  });
  ASSERT_NE(compress, nullptr);

  TimingPropagationOptions options;
  options.defaultRequiredTime = 100;
  ASSERT_TRUE(succeeded(
      options.setRequiredTime(network, compress->getResult(0), 0, 2)));

  auto seed = TimingPropagator::run(network, options);
  ASSERT_TRUE(succeeded(seed));
  auto comp0 = network.findValueBit(compress->getResult(0), 0);
  ASSERT_TRUE(comp0.isValid());
  auto *comp0State = seed->getState(comp0);
  ASSERT_TRUE(comp0State && comp0State->hasRequired);
  EXPECT_EQ(comp0State->required, 2);

  DatapathTimingSemanticsProvider provider;
  ASSERT_TRUE(succeeded(network.refineDynamicSemantics(provider, options)));

  auto path = reconstructCriticalPath(network, options);
  ASSERT_TRUE(succeeded(path));
  EXPECT_EQ(path->delay, 3);

  bool sawFastArc = false;
  for (auto step : path->steps) {
    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    sawFastArc |= arc && arc->token == "compressor_greedy_fast";
  }
  EXPECT_TRUE(sawFastArc);
}

TEST_F(FlatTimingTest, GreedyCompressorCarriesInfluenceNextColumn) {
  auto module = parse(datapathWideCompressIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "wide_compress");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));
  DatapathTimingSemanticsProvider provider;
  ASSERT_TRUE(succeeded(network.refineDynamicSemantics(provider)));

  auto compress = *hwModule.getOps<datapath::CompressOp>().begin();
  auto a0 = network.findValueBit(hwModule.getBodyBlock()->getArgument(0), 0);
  ASSERT_TRUE(a0.isValid());

  bool sawCarryInfluence = false;
  for (auto result : compress->getResults()) {
    auto resultBit1 = network.findValueBit(result, 1);
    ASSERT_TRUE(resultBit1.isValid());
    for (auto arcIndex : network.getPoint(resultBit1)->fanin) {
      auto *arc = network.getArc(arcIndex);
      sawCarryInfluence |= arc && arc->from == a0 &&
                           StringRef(arc->token).starts_with("compressor_greedy_");
    }
  }
  EXPECT_TRUE(sawCarryInfluence);
}

TEST_F(FlatTimingTest, GreedyCompressorUsesFourToTwoStage) {
  auto module = parse(datapathFourToTwoCompressIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "compress42");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));
  DatapathTimingSemanticsProvider provider;
  ASSERT_TRUE(succeeded(network.refineDynamicSemantics(provider)));

  auto compress = *hwModule.getOps<datapath::CompressOp>().begin();
  auto sumBit0 = network.findValueBit(compress->getResult(0), 0);
  ASSERT_TRUE(sumBit0.isValid());

  unsigned compressorInputs = 0;
  for (auto arcIndex : network.getPoint(sumBit0)->fanin) {
    auto *arc = network.getArc(arcIndex);
    if (!arc || arc->token != "compressor_greedy_fast")
      continue;
    EXPECT_EQ(arc->delay, 2);
    ++compressorInputs;
  }
  EXPECT_EQ(compressorInputs, 4u);
}

TEST_F(FlatTimingTest, DatapathSemanticsCompactsPartialProductArcs) {
  auto module = parse(partialProductIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "partial_product");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));
  DatapathTimingSemanticsProvider provider;
  ASSERT_TRUE(succeeded(network.refineDynamicSemantics(provider)));

  auto path = reconstructCriticalPath(network);
  ASSERT_TRUE(succeeded(path));

  bool sawPartialProductArc = false;
  for (auto step : path->steps) {
    if (step.incomingArc == UINT32_MAX)
      continue;
    auto *arc = network.getArc(step.incomingArc);
    sawPartialProductArc |= arc && arc->token == "partial_product" &&
                            arc->kind == TimingArcKind::Synthetic;
  }
  EXPECT_TRUE(sawPartialProductArc);
}

TEST_F(FlatTimingTest, DefaultSemanticsLowersUnknownOpsToDataArcs) {
  auto module = parse(unknownOpIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "unknown");
  ASSERT_TRUE(hwModule);

  TimingNetwork network;
  ASSERT_TRUE(succeeded(network.build(hwModule)));

  Operation *unknown = nullptr;
  hwModule.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "test.unknown_comb")
      unknown = op;
  });
  ASSERT_NE(unknown, nullptr);

  auto result0 = network.findValueBit(unknown->getResult(0), 0);
  ASSERT_TRUE(result0.isValid());

  bool sawGenericDataArc = false;
  for (auto arcIndex : network.getPoint(result0)->fanin) {
    auto *arc = network.getArc(arcIndex);
    sawGenericDataArc |= arc && arc->op == unknown &&
                         arc->kind == TimingArcKind::Data &&
                         arc->token == "op";
  }
  EXPECT_TRUE(sawGenericDataArc);
}

TEST_F(FlatTimingTest, RepairSessionRecordsAndRepairsEdits) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingRepairSession session(hwModule.getOperation());
  ASSERT_TRUE(succeeded(session.initialize()));
  ASSERT_TRUE(session.isInitialized());
  ASSERT_FALSE(session.hasPendingChanges());
  auto *initialNetwork = session.getNetwork();
  ASSERT_NE(initialNetwork, nullptr);
  auto initialArcCount = initialNetwork->getNumArcs();

  auto andOp = *hwModule.getOps<comb::AndOp>().begin();
  session.notifyOperationModified(andOp.getOperation());
  EXPECT_TRUE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());

  ASSERT_TRUE(succeeded(session.repair()));
  EXPECT_FALSE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());
  EXPECT_EQ(session.getNetwork(), initialNetwork);
  EXPECT_EQ(session.getNetwork()->getNumArcs(), initialArcCount);
}

TEST_F(FlatTimingTest, RepairSessionRepairsReplacementLocally) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingRepairSession session(hwModule.getOperation());
  ASSERT_TRUE(succeeded(session.initialize()));
  auto *initialNetwork = session.getNetwork();
  ASSERT_NE(initialNetwork, nullptr);

  auto andOp = *hwModule.getOps<comb::AndOp>().begin();
  auto user = *andOp->user_begin();
  ASSERT_TRUE(isa<comb::XorOp>(user));

  OpBuilder builder(andOp);
  auto replacement =
      comb::OrOp::create(builder, andOp.getLoc(),
                         ValueRange{andOp->getOperand(0), andOp->getOperand(1)},
                         /*twoState=*/true);
  session.notifyOperationInserted(replacement.getOperation(), {});
  session.notifyOperationReplaced(andOp.getOperation(),
                                  ValueRange{replacement.getResult()});
  andOp.getResult().replaceAllUsesWith(replacement.getResult());
  session.notifyOperationErased(andOp.getOperation());
  andOp->erase();

  EXPECT_TRUE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());
  ASSERT_TRUE(succeeded(session.repair()));
  EXPECT_EQ(session.getNetwork(), initialNetwork);
  EXPECT_FALSE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());

  auto replacementPoint =
      session.getNetwork()->findValueBit(replacement.getResult(), 0);
  ASSERT_TRUE(replacementPoint.isValid());
  auto xorOp = cast<comb::XorOp>(user);
  auto xorPoint = session.getNetwork()->findValueBit(xorOp.getResult(), 0);
  ASSERT_TRUE(xorPoint.isValid());

  bool sawReplacementFanin = false;
  for (auto arcIndex : session.getNetwork()->getPoint(xorPoint)->fanin) {
    auto *arc = session.getNetwork()->getArc(arcIndex);
    sawReplacementFanin |=
        arc && arc->from == replacementPoint && arc->op == xorOp.getOperation();
  }
  EXPECT_TRUE(sawReplacementFanin);
}

TEST_F(FlatTimingTest, RepairSessionRepairsDatapathToCombStyleBatch) {
  auto module = parse(partialProductIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "partial_product");
  ASSERT_TRUE(hwModule);

  TimingRepairSession session(hwModule.getOperation());
  ASSERT_TRUE(succeeded(session.initialize()));
  auto *initialNetwork = session.getNetwork();
  ASSERT_NE(initialNetwork, nullptr);

  auto pp = *hwModule.getOps<datapath::PartialProductOp>().begin();
  OpBuilder builder(pp);
  auto repl0 = comb::AndOp::create(builder, pp.getLoc(),
                                   ValueRange{pp.getLhs(), pp.getRhs()},
                                   /*twoState=*/true);
  auto repl1 = comb::OrOp::create(builder, pp.getLoc(),
                                  ValueRange{pp.getLhs(), pp.getRhs()},
                                  /*twoState=*/true);

  session.notifyOperationInserted(repl0.getOperation(), {});
  session.notifyOperationInserted(repl1.getOperation(), {});
  session.notifyOperationReplaced(
      pp.getOperation(), ValueRange{repl0.getResult(), repl1.getResult()});
  pp.getResult(0).replaceAllUsesWith(repl0.getResult());
  pp.getResult(1).replaceAllUsesWith(repl1.getResult());
  session.notifyOperationErased(pp.getOperation());
  pp->erase();

  EXPECT_TRUE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());
  ASSERT_TRUE(succeeded(session.repair()));
  EXPECT_EQ(session.getNetwork(), initialNetwork);
  EXPECT_FALSE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());

  auto path = reconstructCriticalPath(*session.getNetwork());
  ASSERT_TRUE(succeeded(path));
  EXPECT_GT(path->delay, 0);
}

TEST_F(FlatTimingTest, RepairSessionRepairsBoundaryEditsLocally) {
  auto module = parse(propagationIR);
  ASSERT_TRUE(module);
  auto hwModule = getModule(*module, "prop");
  ASSERT_TRUE(hwModule);

  TimingRepairSession session(hwModule.getOperation());
  ASSERT_TRUE(succeeded(session.initialize()));
  auto *initialNetwork = session.getNetwork();
  ASSERT_NE(initialNetwork, nullptr);

  auto output = *hwModule.getOps<hw::OutputOp>().begin();
  session.notifyOperationModified(output.getOperation());
  EXPECT_TRUE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());

  ASSERT_TRUE(succeeded(session.repair()));
  EXPECT_EQ(session.getNetwork(), initialNetwork);
  EXPECT_FALSE(session.hasPendingChanges());
  EXPECT_FALSE(session.needsFullRebuild());
  ASSERT_NE(session.getNetwork(), nullptr);
  EXPECT_GT(session.getNetwork()->getNumArcs(), 0u);
}

} // namespace
