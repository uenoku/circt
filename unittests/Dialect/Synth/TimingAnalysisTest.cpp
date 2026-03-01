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

#include "circt/Dialect/Synth/Analysis/Timing/TimingAnalysis.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Synth/Analysis/Timing/DelayModel.h"
#include "circt/Dialect/Synth/Analysis/Timing/Liberty.h"
#include "circt/Dialect/Synth/Analysis/Timing/RequiredTimeAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>

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

// IR with hierarchy and replicated instances.
const char *hierarchicalIR = R"MLIR(
    hw.module private @leaf(in %clock : !seq.clock, in %a : i1, out x : i1) {
      %r = seq.firreg %a clock %clock : i1
      %c = synth.aig.and_inv %r, %r : i1
      hw.output %c : i1
    }

    hw.module @top(in %clock : !seq.clock, in %a : i1, out x : i1) {
      %x1 = hw.instance "inst1" @leaf(clock: %clock: !seq.clock, a: %a: i1) -> (x: i1)
      %x2 = hw.instance "inst2" @leaf(clock: %clock: !seq.clock, a: %a: i1) -> (x: i1)
      %sum = comb.xor %x1, %x2 : i1
      hw.output %sum : i1
    }
)MLIR";

const char *arcDelayAttrIR = R"MLIR(
    hw.module private @arc_delay_attr(in %a : i1, in %b : i1, out x : i1) {
      %y = comb.and %a, %b {synth.liberty.arc_delay_ps = {i0_o0 = 9 : i64, i1_o0 = 2 : i64}} : i1
      hw.output %y : i1
    }
)MLIR";

const char *libertyBridgeIR = R"MLIR(
    hw.module private @INV(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.006 : f64}},
      out Y : i1 {synth.liberty.pin = {direction = "output"}}) {
      hw.output %A : i1
    }

    hw.module @dut(in %a : i1, out y : i1) {
      %0 = hw.instance "u_inv" @INV(A: %a: i1) -> (Y: i1)
      hw.output %0 : i1
    }
)MLIR";

const char *libertyTimingArcIR = R"MLIR(
    hw.module private @NAND2(
      in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.004 : f64}},
      in %B : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.005 : f64}},
      out Y : i1 {synth.liberty.pin = {
        direction = "output",
        synth.nldm.arcs = [
          #synth.nldm_arc<"A", "Y", "negative_unate", [], [], [0.012 : f64], [], [], [], [], [], [], [], [], []>,
          #synth.nldm_arc<"B", "Y", "negative_unate", [], [], [], [], [], [0.013 : f64], [], [], [], [], [], []>
        ]
      }}) {
      %0 = comb.and %A, %B : i1
      %1 = comb.xor %0, %0 : i1
      hw.output %1 : i1
    }
)MLIR";

const char *nldmCellMapIR = R"MLIR(
    module attributes {synth.liberty.library = {name = "dummy"}} {
      hw.module private @INV(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.006 : f64}},
        out Y : i1 {synth.liberty.pin = {direction = "output"}}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %0 = hw.instance "u_inv" @INV(A: %a: i1) -> (Y: i1) {
          synth.liberty.cell = "INV",
          synth.liberty.arc_delay_ps = {A_to_Y = 11 : i64}
        }
        hw.output %0 : i1
      }
    }
)MLIR";

const char *nldmTimingArcTableIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.002 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.01 : f64, 0.03 : f64], [0.0 : f64, 0.2 : f64], [0.012 : f64, 0.018 : f64, 0.020 : f64, 0.030 : f64], [], [], [], [], [], [], [], [], []>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %0 = hw.instance "u_buf" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %0 : i1
      }
    }
)MLIR";

const char *slewPropagationIR = R"MLIR(
    hw.module @slew_chain(in %a : i1, out y : i1) {
      %0 = comb.and %a, %a : i1
      %1 = comb.and %0, %a : i1
      hw.output %1 : i1
    }
)MLIR";

const char *loadPropagationIR = R"MLIR(
    hw.module @load_tree(in %a : i1, out y : i1) {
      %n0 = comb.and %a, %a : i1
      %n1 = comb.and %n0, %a : i1
      %n2 = comb.and %n0, %a : i1
      %y0 = comb.or %n1, %n2 : i1
      hw.output %y0 : i1
    }
)MLIR";

const char *legacyTimingOnlyIR = R"MLIR(
    module attributes {synth.liberty.library = {name = "dummy", time_unit = "1ns"}} {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.002 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          timing = [{related_pin = "A", cell_rise = [{values = "0.123"}]}]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %0 = hw.instance "u_buf" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %0 : i1
      }
    }
)MLIR";

const char *nldmInterpolationIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64], [0.01 : f64, 0.02 : f64, 0.03 : f64, 0.04 : f64], [], [], [], [], [], [], [], [], []>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        %s1 = hw.instance "u1" @BUF(A: %s0: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %s1 : i1
      }
    }
)MLIR";

const char *nldmTransitionSlewIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64], [0.01 : f64, 0.02 : f64, 0.03 : f64, 0.04 : f64], [], [], [], [0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64], [0.10 : f64, 0.20 : f64, 0.30 : f64, 0.40 : f64], [], [], []>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %s0 : i1
      }
    }
)MLIR";

const char *ccsPilotWaveformIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64], [0.01 : f64, 0.02 : f64, 0.03 : f64, 0.04 : f64], [], [], [], [], [], [], [], [], []>
          ],
          synth.ccs.pilot.arcs = [
            #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.4 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.4 : f64], [1.0 : f64, 0.0 : f64]>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %s0 : i1
      }
    }
)MLIR";

const char *ccsPilotReceiverIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>
          ],
          synth.ccs.pilot.arcs = [
            #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.4 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.4 : f64], [1.0 : f64, 0.0 : f64]>
          ],
          synth.ccs.pilot.receivers = [
            #synth.ccs_pilot_receiver<"A", "Y", [0.0 : f64, 1.0 : f64], [0.0 : f64], [0.2 : f64, 1.2 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64], [0.2 : f64, 1.2 : f64], [], [], [], [], [], []>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %s0 : i1
      }
    }
)MLIR";

const char *ccsPilotWaveformDelayIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>,
      synth.ccs.pilot.waveform_delay = true
    } {
      hw.module private @BUF(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64, 1.0 : f64], [0.0 : f64, 1.0 : f64], [0.01 : f64, 0.02 : f64, 0.03 : f64, 0.04 : f64], [], [], [], [], [], [], [], [], []>
          ],
          synth.ccs.pilot.arcs = [
            #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.4 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.4 : f64], [1.0 : f64, 0.0 : f64]>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF"}
        hw.output %s0 : i1
      }
    }
)MLIR";

const char *ccsPilotMultiInputIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>,
      synth.ccs.pilot.waveform_delay = true
    } {
      hw.module private @AO2(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.2 : f64}},
        in %B : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.2 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>,
            #synth.nldm_arc<"B", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>
          ],
          synth.ccs.pilot.arcs = [
            #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.2 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.2 : f64], [1.0 : f64, 0.0 : f64]>,
            #synth.ccs_pilot_arc<"B", "Y", [0.0 : f64, 0.8 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.8 : f64], [1.0 : f64, 0.0 : f64]>
          ]
        }}) {
        %y = comb.or %A, %B : i1
        hw.output %y : i1
      }

      hw.module @dut(in %a : i1, in %b : i1, out y : i1) {
        %i0 = hw.instance "u0" @AO2(A: %a: i1, B: %b: i1) -> (Y: i1) {synth.liberty.cell = "AO2"}
        hw.output %i0 : i1
      }
    }
)MLIR";

const char *mixedCcsPilotIR = R"MLIR(
    module attributes {
      synth.liberty.library = {name = "dummy", time_unit = "1ns"},
      synth.nldm.time_unit = #synth.nldm_time_unit<1000.0>,
      synth.ccs.pilot.waveform_delay = true,
      synth.ccs.pilot.cells = ["BUF_CCS"]
    } {
      hw.module private @BUF_CCS(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>
          ],
          synth.ccs.pilot.arcs = [
            #synth.ccs_pilot_arc<"A", "Y", [0.0 : f64, 0.4 : f64], [0.0 : f64, 1.0 : f64], [0.0 : f64, 0.4 : f64], [1.0 : f64, 0.0 : f64]>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module private @BUF_NLDM(
        in %A : i1 {synth.liberty.pin = {direction = "input", capacitance = 0.5 : f64}},
        out Y : i1 {synth.liberty.pin = {
          direction = "output",
          synth.nldm.arcs = [
            #synth.nldm_arc<"A", "Y", "positive_unate", [0.0 : f64], [0.0 : f64], [0.025 : f64], [], [], [], [], [], [], [], [], []>
          ]
        }}) {
        hw.output %A : i1
      }

      hw.module @dut(in %a : i1, out y : i1) {
        %s0 = hw.instance "u0" @BUF_CCS(A: %a: i1) -> (Y: i1) {synth.liberty.cell = "BUF_CCS"}
        %s1 = hw.instance "u1" @BUF_NLDM(A: %s0: i1) -> (Y: i1) {synth.liberty.cell = "BUF_NLDM"}
        hw.output %s1 : i1
      }
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

class SlewTestDelayModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    return {1, ctx.inputSlew + 0.25};
  }

  llvm::StringRef getName() const override { return "slew-test"; }
  bool usesSlewPropagation() const override { return true; }
};

class LoadTestDelayModel final : public DelayModel {
public:
  explicit LoadTestDelayModel(double capPerInput) : capPerInput(capPerInput) {}

  DelayResult computeDelay(const DelayContext &ctx) const override {
    (void)ctx;
    return {static_cast<int64_t>(std::llround(ctx.outputLoad * 100.0)),
            ctx.inputSlew};
  }

  llvm::StringRef getName() const override { return "load-test"; }

  double getInputCapacitance(const DelayContext &ctx) const override {
    (void)ctx;
    return capPerInput;
  }

private:
  double capPerInput;
};

class FixedSlewDelayModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    return {1, ctx.inputSlew};
  }

  llvm::StringRef getName() const override { return "fixed-slew"; }
  bool usesSlewPropagation() const override { return true; }
};

class CoupledLoadSlewModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    int64_t delay = static_cast<int64_t>(std::llround(ctx.outputLoad * 100.0));
    return {delay, ctx.inputSlew + 0.2};
  }

  double getInputCapacitance(const DelayContext &ctx) const override {
    return 0.5 + 0.5 * ctx.inputSlew;
  }

  llvm::StringRef getName() const override { return "coupled-load-slew"; }
  bool usesSlewPropagation() const override { return true; }
};

class OscillatingHintModel final : public DelayModel {
public:
  DelayResult computeDelay(const DelayContext &ctx) const override {
    int64_t delay = static_cast<int64_t>(std::llround(ctx.outputLoad * 100.0));
    return {delay, ctx.outputLoad};
  }

  double getInputCapacitance(const DelayContext &ctx) const override {
    return std::max(0.0, 1.0 - ctx.inputSlew);
  }

  llvm::StringRef getName() const override { return "oscillating-hint"; }
  bool usesSlewPropagation() const override { return true; }
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

TEST_F(TimingAnalysisTest, ArrivalAnalysisPropagatesSlewWhenEnabled) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(slewPropagationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("slew_chain");
  ASSERT_TRUE(hwModule);

  SlewTestDelayModel slewModel;
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(&slewModel)));

  ArrivalAnalysis::Options opts;
  opts.keepAllArrivals = true;
  opts.initialSlew = 0.5;

  ArrivalAnalysis arrivals(graph, opts, &slewModel);
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  auto infos = arrivals.getArrivals(endPoint->getId());
  ASSERT_FALSE(infos.empty());
  EXPECT_GT(infos.front().slew, 0.5);
}

TEST_F(TimingAnalysisTest, ArrivalAnalysisPropagatesOutputLoadToDelayModel) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(loadPropagationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("load_tree");
  ASSERT_TRUE(hwModule);

  LoadTestDelayModel zeroCap(0.0);
  TimingGraph graphZero(hwModule);
  ASSERT_TRUE(succeeded(graphZero.build(&zeroCap)));
  ArrivalAnalysis arrivalsZero(graphZero, {}, &zeroCap);
  ASSERT_TRUE(succeeded(arrivalsZero.run()));
  ASSERT_EQ(graphZero.getEndPoints().size(), 1u);
  int64_t zeroDelay =
      arrivalsZero.getMaxArrivalTime(graphZero.getEndPoints().front());

  LoadTestDelayModel nonZeroCap(0.5);
  TimingGraph graphLoad(hwModule);
  ASSERT_TRUE(succeeded(graphLoad.build(&nonZeroCap)));
  ArrivalAnalysis arrivalsLoad(graphLoad, {}, &nonZeroCap);
  ASSERT_TRUE(succeeded(arrivalsLoad.run()));
  ASSERT_EQ(graphLoad.getEndPoints().size(), 1u);
  int64_t loadedDelay =
      arrivalsLoad.getMaxArrivalTime(graphLoad.getEndPoints().front());

  EXPECT_GT(loadedDelay, zeroDelay);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelPerArcAttr) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(arcDelayAttrIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("arc_delay_attr");
  ASSERT_TRUE(hwModule);

  NLDMDelayModel nldmModel;
  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(&nldmModel)));

  ArrivalAnalysis::Options opts;
  opts.keepAllArrivals = true;
  ArrivalAnalysis arrivals(graph, opts, &nldmModel);
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();

  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 9);

  auto infos = arrivals.getArrivals(endPoint->getId());
  ASSERT_EQ(infos.size(), 2u);
  int64_t minArrival = std::min(infos[0].arrivalTime, infos[1].arrivalTime);
  int64_t maxArrival = std::max(infos[0].arrivalTime, infos[1].arrivalTime);
  EXPECT_EQ(minArrival, 2);
  EXPECT_EQ(maxArrival, 9);
}

TEST_F(TimingAnalysisTest, LibertyBridgePinCapacitance) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(libertyBridgeIR, &context);
  ASSERT_TRUE(module);

  auto libOr = LibertyLibrary::fromModule(module.get());
  ASSERT_FALSE(failed(libOr));
  LibertyLibrary lib = *libOr;

  auto invInCap = lib.getInputPinCapacitance("INV", "A");
  ASSERT_TRUE(invInCap.has_value());
  EXPECT_NEAR(*invInCap, 0.006, 1e-12);

  auto invOutCap = lib.getInputPinCapacitance("INV", "Y");
  EXPECT_FALSE(invOutCap.has_value());

  auto inPin0 = lib.getInputPinName("INV", 0);
  ASSERT_TRUE(inPin0.has_value());
  EXPECT_EQ(*inPin0, "A");

  auto outPin0 = lib.getOutputPinName("INV", 0);
  ASSERT_TRUE(outPin0.has_value());
  EXPECT_EQ(*outPin0, "Y");

  auto capByIndex = lib.getInputPinCapacitance("INV", 0);
  ASSERT_TRUE(capByIndex.has_value());
  EXPECT_NEAR(*capByIndex, 0.006, 1e-12);
}

TEST_F(TimingAnalysisTest, LibertyBridgeTimingArcLookup) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(libertyTimingArcIR, &context);
  ASSERT_TRUE(module);

  auto libOr = LibertyLibrary::fromModule(module.get());
  ASSERT_FALSE(failed(libOr));
  LibertyLibrary lib = *libOr;

  auto byNameA = lib.getTimingArc("NAND2", "A", "Y");
  ASSERT_TRUE(byNameA.has_value());
  EXPECT_TRUE(byNameA->getAs<StringAttr>("related_pin"));

  auto byNameB = lib.getTimingArc("NAND2", "B", "Y");
  ASSERT_TRUE(byNameB.has_value());
  auto relatedPinB = byNameB->getAs<StringAttr>("related_pin");
  ASSERT_TRUE(relatedPinB);
  EXPECT_EQ(relatedPinB.getValue(), "B");

  auto byIndex = lib.getTimingArc("NAND2", 1, 0);
  ASSERT_TRUE(byIndex.has_value());
  EXPECT_TRUE(byIndex->getAs<StringAttr>("timing_sense"));

  auto typed = lib.getTypedTimingArc("NAND2", 1, 0);
  ASSERT_TRUE(typed.has_value());
  EXPECT_EQ(typed->getRelatedPin().getValue(), "B");
}

TEST_F(TimingAnalysisTest, LibertyBridgeTimingArcLookupPrefersNldmArcs) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmTimingArcTableIR, &context);
  ASSERT_TRUE(module);

  auto libOr = LibertyLibrary::fromModule(module.get());
  ASSERT_FALSE(failed(libOr));
  LibertyLibrary lib = *libOr;

  auto arc = lib.getTimingArc("BUF", "A", "Y");
  ASSERT_TRUE(arc.has_value());

  auto rise = dyn_cast_or_null<ArrayAttr>(arc->get("cell_rise_values"));
  ASSERT_TRUE(rise);
  ASSERT_FALSE(rise.empty());
  auto v0 = dyn_cast<FloatAttr>(rise[0]);
  ASSERT_TRUE(v0);
  EXPECT_NEAR(v0.getValueAsDouble(), 0.012, 1e-12);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelResolvesCellPinMapping) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmCellMapIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto delayModel = createNLDMDelayModel(module.get());
  ASSERT_NE(delayModel, nullptr);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(delayModel.get())));

  ArrivalAnalysis::Options opts;
  opts.keepAllArrivals = true;
  ArrivalAnalysis arrivals(graph, opts, delayModel.get());
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 11);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelReadsTimingArcTableValue) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmTimingArcTableIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto delayModel = createNLDMDelayModel(module.get());
  ASSERT_NE(delayModel, nullptr);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(delayModel.get())));

  ArrivalAnalysis arrivals(graph, {}, delayModel.get());
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 12);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelIgnoresLegacyTimingGroups) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(legacyTimingOnlyIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto delayModel = createNLDMDelayModel(module.get());
  ASSERT_NE(delayModel, nullptr);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(delayModel.get())));

  ArrivalAnalysis arrivals(graph, {}, delayModel.get());
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 0);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelInterpolatesOverSlewAndLoad) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmInterpolationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto delayModel = createNLDMDelayModel(module.get());
  ASSERT_NE(delayModel, nullptr);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(delayModel.get())));

  ArrivalAnalysis::Options opts;
  opts.initialSlew = 0.5;
  ArrivalAnalysis arrivals(graph, opts, delayModel.get());
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 50);
}

TEST_F(TimingAnalysisTest, NLDMDelayModelInterpolatesOutputSlew) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmTransitionSlewIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto delayModel = createNLDMDelayModel(module.get());
  ASSERT_NE(delayModel, nullptr);

  TimingGraph graph(hwModule);
  ASSERT_TRUE(succeeded(graph.build(delayModel.get())));

  ArrivalAnalysis::Options opts;
  opts.initialSlew = 0.5;
  opts.keepAllArrivals = true;
  ArrivalAnalysis arrivals(graph, opts, delayModel.get());
  ASSERT_TRUE(succeeded(arrivals.run()));

  ASSERT_EQ(graph.getEndPoints().size(), 1u);
  auto *endPoint = graph.getEndPoints().front();
  EXPECT_EQ(arrivals.getMaxArrivalTime(endPoint), 25);

  auto infos = arrivals.getArrivals(endPoint->getId());
  ASSERT_FALSE(infos.empty());
  EXPECT_NEAR(infos.front().slew, 0.25, 1e-9);
}

TEST_F(TimingAnalysisTest, CCSPilotDelayModelProducesWaveform) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->getName(), "ccs-pilot");
  EXPECT_TRUE(model->usesSlewPropagation());
  EXPECT_TRUE(model->usesWaveformPropagation());

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext ctx;
  ctx.op = inst;
  ctx.inputValue = inst.getOperand(0);
  ctx.outputValue = inst.getResult(0);
  ctx.inputIndex = 0;
  ctx.outputIndex = 0;
  ctx.inputSlew = 0.5;
  ctx.outputLoad = 0.5;

  auto delay = model->computeDelay(ctx);
  EXPECT_GT(delay.delay, 0);

  SmallVector<WaveformPoint> inputWaveform = {{0.0, 0.0}, {1.0, 1.0}};
  SmallVector<WaveformPoint> outputWaveform;
  EXPECT_TRUE(model->computeOutputWaveform(ctx, inputWaveform, outputWaveform));
  ASSERT_EQ(outputWaveform.size(), 2u);
  EXPECT_NEAR(outputWaveform[0].time, 25.0, 1e-9);
  EXPECT_NEAR(outputWaveform[1].time, 425.0, 1e-9);
  EXPECT_EQ(outputWaveform[0].value, 0.0);
  EXPECT_EQ(outputWaveform[1].value, 1.0);
}

TEST_F(TimingAnalysisTest, CCSPilotDelayModelUsesFallWaveformForFallingEdge) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext ctx;
  ctx.op = inst;
  ctx.inputValue = inst.getOperand(0);
  ctx.outputValue = inst.getResult(0);
  ctx.inputIndex = 0;
  ctx.outputIndex = 0;
  ctx.inputSlew = 0.5;
  ctx.outputLoad = 0.5;

  SmallVector<WaveformPoint> inputWaveform = {{0.0, 1.0}, {1.0, 0.0}};
  SmallVector<WaveformPoint> outputWaveform;
  EXPECT_TRUE(model->computeOutputWaveform(ctx, inputWaveform, outputWaveform));
  ASSERT_EQ(outputWaveform.size(), 2u);
  EXPECT_NEAR(outputWaveform[0].time, 25.0, 1e-9);
  EXPECT_NEAR(outputWaveform[1].time, 425.0, 1e-9);
  EXPECT_EQ(outputWaveform[0].value, 1.0);
  EXPECT_EQ(outputWaveform[1].value, 0.0);
}

TEST_F(TimingAnalysisTest, CCSPilotWaveformStretchesWithOutputLoad) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext lowLoad;
  lowLoad.op = inst;
  lowLoad.inputValue = inst.getOperand(0);
  lowLoad.outputValue = inst.getResult(0);
  lowLoad.inputIndex = 0;
  lowLoad.outputIndex = 0;
  lowLoad.inputSlew = 0.5;
  lowLoad.outputLoad = 0.5;

  DelayContext highLoad = lowLoad;
  highLoad.outputLoad = 1.5;

  SmallVector<WaveformPoint> inputWaveform = {{0.0, 0.0}, {1.0, 1.0}};
  SmallVector<WaveformPoint> lowWaveform;
  SmallVector<WaveformPoint> highWaveform;
  EXPECT_TRUE(
      model->computeOutputWaveform(lowLoad, inputWaveform, lowWaveform));
  EXPECT_TRUE(
      model->computeOutputWaveform(highLoad, inputWaveform, highWaveform));
  ASSERT_EQ(lowWaveform.size(), 2u);
  ASSERT_EQ(highWaveform.size(), 2u);
  EXPECT_LT(lowWaveform[1].time, highWaveform[1].time);
}

TEST_F(TimingAnalysisTest, CCSPilotReceiverDataAffectsWaveformStretch) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotReceiverIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext lowSlew;
  lowSlew.op = inst;
  lowSlew.inputValue = inst.getOperand(0);
  lowSlew.outputValue = inst.getResult(0);
  lowSlew.inputIndex = 0;
  lowSlew.outputIndex = 0;
  lowSlew.inputSlew = 0.0;
  lowSlew.outputLoad = 0.5;

  DelayContext highSlew = lowSlew;
  highSlew.inputSlew = 1.0;

  SmallVector<WaveformPoint> inputWaveform = {{0.0, 0.0}, {1.0, 1.0}};
  SmallVector<WaveformPoint> lowWaveform;
  SmallVector<WaveformPoint> highWaveform;
  EXPECT_TRUE(
      model->computeOutputWaveform(lowSlew, inputWaveform, lowWaveform));
  EXPECT_TRUE(
      model->computeOutputWaveform(highSlew, inputWaveform, highWaveform));
  ASSERT_EQ(lowWaveform.size(), 2u);
  ASSERT_EQ(highWaveform.size(), 2u);
  EXPECT_LT(lowWaveform[1].time, highWaveform[1].time);
}

TEST_F(TimingAnalysisTest, CCSPilotDelayUsesWaveformThresholdWhenEnabled) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformDelayIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext ctx;
  ctx.op = inst;
  ctx.inputValue = inst.getOperand(0);
  ctx.outputValue = inst.getResult(0);
  ctx.inputIndex = 0;
  ctx.outputIndex = 0;
  ctx.inputSlew = 0.5;
  ctx.outputLoad = 0.5;

  auto delay = model->computeDelay(ctx);
  EXPECT_EQ(delay.delay, 200);
  EXPECT_NEAR(delay.outputSlew, 320.0, 1e-9);
}

TEST_F(TimingAnalysisTest, CCSPilotDelayFallsBackToNLDMDelayWhenDisabled) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext ctx;
  ctx.op = inst;
  ctx.inputValue = inst.getOperand(0);
  ctx.outputValue = inst.getResult(0);
  ctx.inputIndex = 0;
  ctx.outputIndex = 0;
  ctx.inputSlew = 0.5;
  ctx.outputLoad = 0.5;

  auto delay = model->computeDelay(ctx);
  EXPECT_EQ(delay.delay, 25);
  EXPECT_NEAR(delay.outputSlew, 320.0, 1e-9);
}

TEST_F(TimingAnalysisTest, CCSPilotMultiInputArcsProduceDifferentDelay) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotMultiInputIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  auto inst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(inst);

  DelayContext aCtx;
  aCtx.op = inst;
  aCtx.inputValue = inst.getOperand(0);
  aCtx.outputValue = inst.getResult(0);
  aCtx.inputIndex = 0;
  aCtx.outputIndex = 0;
  aCtx.inputSlew = 0.5;
  aCtx.outputLoad = 0.5;

  DelayContext bCtx = aCtx;
  bCtx.inputValue = inst.getOperand(1);
  bCtx.inputIndex = 1;

  auto aDelay = model->computeDelay(aCtx);
  auto bDelay = model->computeDelay(bCtx);
  EXPECT_EQ(aDelay.delay, 100);
  EXPECT_EQ(bDelay.delay, 400);
  EXPECT_LT(aDelay.delay, bDelay.delay);
}

TEST_F(TimingAnalysisTest, MixedCCSPilotDelegatesByCellPolicy) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(mixedCcsPilotIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createMixedNLDMCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->getName(), "mixed-ccs-pilot");

  auto firstInst = dyn_cast<hw::InstanceOp>(&hwModule.getBodyBlock()->front());
  ASSERT_TRUE(firstInst);
  auto secondInst = dyn_cast<hw::InstanceOp>(firstInst->getNextNode());
  ASSERT_TRUE(secondInst);

  DelayContext ccsCtx;
  ccsCtx.op = firstInst;
  ccsCtx.inputValue = firstInst.getOperand(0);
  ccsCtx.outputValue = firstInst.getResult(0);
  ccsCtx.inputIndex = 0;
  ccsCtx.outputIndex = 0;
  ccsCtx.inputSlew = 0.5;
  ccsCtx.outputLoad = 0.5;

  DelayContext nldmCtx;
  nldmCtx.op = secondInst;
  nldmCtx.inputValue = secondInst.getOperand(0);
  nldmCtx.outputValue = secondInst.getResult(0);
  nldmCtx.inputIndex = 0;
  nldmCtx.outputIndex = 0;
  nldmCtx.inputSlew = 0.5;
  nldmCtx.outputLoad = 0.5;

  auto ccsDelay = model->computeDelay(ccsCtx);
  auto nldmDelay = model->computeDelay(nldmCtx);
  EXPECT_EQ(ccsDelay.delay, 200);
  EXPECT_EQ(nldmDelay.delay, 25);
  EXPECT_TRUE(model->usesWaveformPropagation());

  SmallVector<WaveformPoint> inputWaveform = {{0.0, 0.0}, {1.0, 1.0}};
  SmallVector<WaveformPoint> waveform;
  EXPECT_TRUE(model->computeOutputWaveform(ccsCtx, inputWaveform, waveform));
  EXPECT_FALSE(model->computeOutputWaveform(nldmCtx, inputWaveform, waveform));
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
  EXPECT_NE(report.find("Slew Hint Damping:"), std::string::npos);
  EXPECT_NE(report.find("Adaptive Slew Damping Mode:"), std::string::npos);
  EXPECT_NE(report.find("Applied Slew Hint Damping:"), std::string::npos);
  EXPECT_NE(report.find("Max Slew Delta:"), std::string::npos);
  EXPECT_NE(report.find("Relative Max Slew Delta:"), std::string::npos);
  EXPECT_NE(report.find("Relative Slew Epsilon:"), std::string::npos);
  EXPECT_NE(report.find("Effective Relative Slew Epsilon:"), std::string::npos);
  EXPECT_NE(report.find("Waveform-Coupled Convergence:"), std::string::npos);
  EXPECT_NE(report.find("Slew Delta Trend:"), std::string::npos);
  EXPECT_NE(report.find("Slew Trend Class:"), std::string::npos);
  EXPECT_NE(report.find("Slew Reduction Ratio:"), std::string::npos);
  EXPECT_NE(report.find("Slew Advice:"), std::string::npos);
  EXPECT_EQ(report.find("--- Slew Convergence ---"), std::string::npos);
  EXPECT_NE(report.find("Worst Slack:"), std::string::npos);
  EXPECT_NE(report.find("Critical Paths"), std::string::npos);
}

TEST_F(TimingAnalysisTest, ReportTimingIncludesConvergenceTableWhenEnabled) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  FixedSlewDelayModel model;
  TimingAnalysisOptions opts;
  opts.delayModel = &model;
  opts.initialSlew = 0.25;
  opts.emitSlewConvergenceTable = true;

  auto analysis = TimingAnalysis::create(hwModule, opts);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));

  std::string report;
  llvm::raw_string_ostream os(report);
  analysis->reportTiming(os, 5);

  EXPECT_NE(report.find("--- Slew Convergence ---"), std::string::npos);
  EXPECT_NE(report.find("Iter | Max Slew Delta | Applied Damping"),
            std::string::npos);
}

TEST_F(TimingAnalysisTest, ReportTimingIncludesWaveformDetailsWhenEnabled) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmInterpolationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  TimingAnalysisOptions opts;
  opts.delayModel = model.get();
  opts.initialSlew = 0.5;
  opts.keepAllArrivals = true;
  opts.emitWaveformDetails = true;

  auto analysis = TimingAnalysis::create(hwModule, opts);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));

  std::string report;
  llvm::raw_string_ostream os(report);
  analysis->reportTiming(os, 5);

  EXPECT_NE(report.find("Waveform Details:"), std::string::npos);
  EXPECT_NE(report.find("(t="), std::string::npos);
  EXPECT_NE(report.find(", v="), std::string::npos);
  EXPECT_NE(report.find("t50="), std::string::npos);
  EXPECT_NE(report.find("slew10-90="), std::string::npos);
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

TEST_F(TimingAnalysisTest, FullPipelineRunsSlewConvergenceLoop) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(testIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("simple");
  ASSERT_TRUE(hwModule);

  FixedSlewDelayModel model;
  TimingAnalysisOptions opts;
  opts.delayModel = &model;
  opts.initialSlew = 0.25;
  opts.maxSlewIterations = 5;
  opts.slewConvergenceEpsilon = 1e-12;

  auto analysis = TimingAnalysis::create(hwModule, opts);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));
  EXPECT_GE(analysis->getLastArrivalIterations(), 1u);
  EXPECT_LE(analysis->getLastArrivalIterations(), opts.maxSlewIterations);
  EXPECT_TRUE(analysis->didLastArrivalConverge());
  EXPECT_LE(analysis->getLastMaxSlewDelta(), opts.slewConvergenceEpsilon);
  EXPECT_FALSE(analysis->getLastSlewDeltaHistory().empty());
  EXPECT_EQ(analysis->getLastSlewDeltaHistory().size(),
            analysis->getLastSlewDampingHistory().size());
  EXPECT_EQ(analysis->getLastSlewDeltaHistory().back(),
            analysis->getLastMaxSlewDelta());
}

TEST_F(TimingAnalysisTest, FullPipelineDetectsNonConvergence) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(slewPropagationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("slew_chain");
  ASSERT_TRUE(hwModule);

  SlewTestDelayModel model;
  TimingAnalysisOptions opts;
  opts.delayModel = &model;
  opts.initialSlew = 0.5;
  opts.maxSlewIterations = 1;
  opts.slewConvergenceEpsilon = 1e-12;

  auto analysis = TimingAnalysis::create(hwModule, opts);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));
  EXPECT_EQ(analysis->getLastArrivalIterations(), 1u);
  EXPECT_FALSE(analysis->didLastArrivalConverge());
  EXPECT_GT(analysis->getLastMaxSlewDelta(), opts.slewConvergenceEpsilon);
}

TEST_F(TimingAnalysisTest, RelativeSlewConvergenceEpsilonCanTerminateEarly) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(slewPropagationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("slew_chain");
  ASSERT_TRUE(hwModule);

  SlewTestDelayModel model;

  TimingAnalysisOptions absoluteOnlyOpts;
  absoluteOnlyOpts.delayModel = &model;
  absoluteOnlyOpts.initialSlew = 0.0;
  absoluteOnlyOpts.maxSlewIterations = 1;
  absoluteOnlyOpts.slewConvergenceEpsilon = 1e-12;

  auto absoluteOnly = TimingAnalysis::create(hwModule, absoluteOnlyOpts);
  ASSERT_NE(absoluteOnly, nullptr);
  ASSERT_TRUE(succeeded(absoluteOnly->runFullAnalysis()));
  EXPECT_FALSE(absoluteOnly->didLastArrivalConverge());

  TimingAnalysisOptions relativeOpts = absoluteOnlyOpts;
  relativeOpts.slewConvergenceRelativeEpsilon = 1.0;

  auto relative = TimingAnalysis::create(hwModule, relativeOpts);
  ASSERT_NE(relative, nullptr);
  ASSERT_TRUE(succeeded(relative->runFullAnalysis()));
  EXPECT_TRUE(relative->didLastArrivalConverge());
  EXPECT_LE(relative->getLastRelativeSlewDelta(),
            relativeOpts.slewConvergenceRelativeEpsilon);
  EXPECT_GT(relative->getLastSlewDeltaHistory().front(), 0.0);
}

TEST_F(TimingAnalysisTest, ConvergenceLoopUpdatesLoadFromSlewHints) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(slewPropagationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("slew_chain");
  ASSERT_TRUE(hwModule);

  CoupledLoadSlewModel model;

  TimingAnalysisOptions oneIterOpts;
  oneIterOpts.delayModel = &model;
  oneIterOpts.initialSlew = 0.0;
  oneIterOpts.maxSlewIterations = 1;
  oneIterOpts.slewConvergenceEpsilon = 1e-12;

  auto oneIter = TimingAnalysis::create(hwModule, oneIterOpts);
  ASSERT_NE(oneIter, nullptr);
  ASSERT_TRUE(succeeded(oneIter->runFullAnalysis()));
  ASSERT_FALSE(oneIter->didLastArrivalConverge());

  auto *endPointOne = oneIter->getGraph().getEndPoints().front();
  int64_t delayOneIter = oneIter->getArrivalTime(endPointOne);

  TimingAnalysisOptions convOpts = oneIterOpts;
  convOpts.maxSlewIterations = 6;
  auto converged = TimingAnalysis::create(hwModule, convOpts);
  ASSERT_NE(converged, nullptr);
  ASSERT_TRUE(succeeded(converged->runFullAnalysis()));
  ASSERT_TRUE(converged->didLastArrivalConverge());

  auto *endPointConv = converged->getGraph().getEndPoints().front();
  int64_t delayConverged = converged->getArrivalTime(endPointConv);

  EXPECT_GT(delayConverged, delayOneIter);
}

TEST_F(TimingAnalysisTest, SlewHintDampingChangesConvergenceTrajectory) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmInterpolationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  OscillatingHintModel model;

  TimingAnalysisOptions undampedOpts;
  undampedOpts.delayModel = &model;
  undampedOpts.initialSlew = 0.0;
  undampedOpts.maxSlewIterations = 6;
  undampedOpts.slewConvergenceEpsilon = 1e-9;
  undampedOpts.slewHintDamping = 1.0;

  auto undamped = TimingAnalysis::create(hwModule, undampedOpts);
  ASSERT_NE(undamped, nullptr);
  ASSERT_TRUE(succeeded(undamped->runFullAnalysis()));

  TimingAnalysisOptions dampedOpts = undampedOpts;
  dampedOpts.slewHintDamping = 0.5;
  dampedOpts.maxSlewIterations = 8;

  auto damped = TimingAnalysis::create(hwModule, dampedOpts);
  ASSERT_NE(damped, nullptr);
  ASSERT_TRUE(succeeded(damped->runFullAnalysis()));

  EXPECT_NE(undamped->getLastArrivalIterations(),
            damped->getLastArrivalIterations());
  EXPECT_NE(undamped->getLastMaxSlewDelta(), damped->getLastMaxSlewDelta());
}

TEST_F(TimingAnalysisTest, AdaptiveSlewHintDampingAdjustsAppliedFactor) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmInterpolationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  OscillatingHintModel model;

  TimingAnalysisOptions opts;
  opts.delayModel = &model;
  opts.initialSlew = 0.0;
  opts.maxSlewIterations = 6;
  opts.slewConvergenceEpsilon = 1e-9;
  opts.slewHintDamping = 1.0;
  opts.adaptiveSlewHintDampingMode =
      TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive;

  auto analysis = TimingAnalysis::create(hwModule, opts);
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));

  EXPECT_TRUE(analysis->isAdaptiveSlewHintDampingEnabled());
  EXPECT_EQ(analysis->getConfiguredAdaptiveSlewHintDampingMode(),
            TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive);
  EXPECT_GE(analysis->getLastAppliedSlewHintDamping(), 0.1);
  EXPECT_LE(analysis->getLastAppliedSlewHintDamping(), 1.0);
  EXPECT_FALSE(analysis->getLastSlewDeltaHistory().empty());
}

TEST_F(TimingAnalysisTest, AdaptiveDampingPolicyAffectsAppliedFactor) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(nldmInterpolationIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  OscillatingHintModel model;

  TimingAnalysisOptions conservativeOpts;
  conservativeOpts.delayModel = &model;
  conservativeOpts.initialSlew = 0.0;
  conservativeOpts.maxSlewIterations = 6;
  conservativeOpts.slewConvergenceEpsilon = 1e-9;
  conservativeOpts.slewHintDamping = 1.0;
  conservativeOpts.adaptiveSlewHintDampingMode =
      TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Conservative;

  auto conservative = TimingAnalysis::create(hwModule, conservativeOpts);
  ASSERT_NE(conservative, nullptr);
  ASSERT_TRUE(succeeded(conservative->runFullAnalysis()));

  TimingAnalysisOptions aggressiveOpts = conservativeOpts;
  aggressiveOpts.adaptiveSlewHintDampingMode =
      TimingAnalysisOptions::AdaptiveSlewHintDampingMode::Aggressive;

  auto aggressive = TimingAnalysis::create(hwModule, aggressiveOpts);
  ASSERT_NE(aggressive, nullptr);
  ASSERT_TRUE(succeeded(aggressive->runFullAnalysis()));

  EXPECT_TRUE(aggressive->isAdaptiveSlewHintDampingEnabled());
  EXPECT_TRUE(conservative->isAdaptiveSlewHintDampingEnabled());
  bool differs =
      aggressive->getLastAppliedSlewHintDamping() !=
          conservative->getLastAppliedSlewHintDamping() ||
      aggressive->getLastArrivalIterations() !=
          conservative->getLastArrivalIterations() ||
      aggressive->getLastMaxSlewDelta() != conservative->getLastMaxSlewDelta();
  EXPECT_TRUE(differs);
}

TEST_F(TimingAnalysisTest, WaveformCoupledConvergenceHeuristicsAdjustDefaults) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ccsPilotWaveformIR, &context);
  ASSERT_TRUE(module);

  SymbolTable symbolTable(module.get());
  auto hwModule = symbolTable.lookup<hw::HWModuleOp>("dut");
  ASSERT_TRUE(hwModule);

  auto model = createCCSPilotDelayModel(module.get());
  ASSERT_NE(model, nullptr);

  TimingAnalysisOptions onOpts;
  onOpts.delayModel = model.get();
  onOpts.maxSlewIterations = 1;
  onOpts.slewConvergenceRelativeEpsilon = 0.0;
  onOpts.slewHintDamping = 1.0;
  onOpts.enableWaveformCoupledConvergence = true;

  auto on = TimingAnalysis::create(hwModule, onOpts);
  ASSERT_NE(on, nullptr);
  ASSERT_TRUE(succeeded(on->runFullAnalysis()));
  EXPECT_TRUE(on->usedWaveformCoupledConvergence());
  EXPECT_EQ(on->getLastArrivalIterations(), 2u);
  EXPECT_NEAR(on->getLastEffectiveSlewConvergenceRelativeEpsilon(), 0.05,
              1e-12);
  ASSERT_FALSE(on->getLastSlewDampingHistory().empty());
  EXPECT_NEAR(on->getLastSlewDampingHistory().front(), 0.8, 1e-12);

  TimingAnalysisOptions offOpts = onOpts;
  offOpts.enableWaveformCoupledConvergence = false;

  auto off = TimingAnalysis::create(hwModule, offOpts);
  ASSERT_NE(off, nullptr);
  ASSERT_TRUE(succeeded(off->runFullAnalysis()));
  EXPECT_FALSE(off->usedWaveformCoupledConvergence());
  EXPECT_EQ(off->getLastArrivalIterations(), 1u);
  EXPECT_EQ(off->getLastEffectiveSlewConvergenceRelativeEpsilon(), 0.0);
  ASSERT_FALSE(off->getLastSlewDampingHistory().empty());
  EXPECT_NEAR(off->getLastSlewDampingHistory().front(), 1.0, 1e-12);
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

TEST_F(TimingAnalysisTest, HierarchicalTimingAnalysisRequiresTop) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(hierarchicalIR, &context);
  ASSERT_TRUE(module);

  auto missingTop = TimingAnalysis::create(*module, "");
  EXPECT_EQ(missingTop, nullptr);

  auto badTop = TimingAnalysis::create(*module, "does_not_exist");
  EXPECT_EQ(badTop, nullptr);
}

TEST_F(TimingAnalysisTest, HierarchicalTimingAnalysisElaboratesInstances) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(hierarchicalIR, &context);
  ASSERT_TRUE(module);

  auto analysis = TimingAnalysis::create(*module, "top");
  ASSERT_NE(analysis, nullptr);
  ASSERT_TRUE(succeeded(analysis->runFullAnalysis()));
  EXPECT_GT(analysis->getGraph().getStartPoints().size(), 0u);
  EXPECT_GT(analysis->getGraph().getEndPoints().size(), 0u);
  EXPECT_EQ(analysis->getGraph().getTopologicalOrder().size(),
            analysis->getGraph().getNumNodes());

  SmallVector<TimingPath> paths;
  ASSERT_TRUE(succeeded(analysis->getKWorstPaths(20, paths)));
  ASSERT_FALSE(paths.empty());

  bool sawInst1 = false;
  bool sawInst2 = false;
  for (auto &path : paths) {
    sawInst1 |= path.getStartPoint()->getName().contains("inst1/");
    sawInst2 |= path.getStartPoint()->getName().contains("inst2/");
  }

  EXPECT_TRUE(sawInst1);
  EXPECT_TRUE(sawInst2);
}

} // namespace
