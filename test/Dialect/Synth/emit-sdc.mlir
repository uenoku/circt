// RUN: circt-opt --synth-emit-sdc='top-module-name=top output-file="-" clock-period-ns=5.0' %s | FileCheck %s
// RUN: circt-opt --synth-emit-sdc='top-module-name=top2 output-file="-" clock-gate-cells="eicg_stub:out=in" clock-source-cells="fixedclk"' %s | FileCheck %s --check-prefix=SDC2
// RUN: circt-opt --synth-emit-sdc='top-module-name=top3 output-file="-"' %s | FileCheck %s --check-prefix=SDC3
// RUN: circt-opt --synth-emit-sdc='top-module-name=top4 output-file="-" clock-period-ns=5.0 clock-uncertainty-setup-ns=0.05 clock-uncertainty-hold-ns=0.02 reset-ports="rst"' %s | FileCheck %s --check-prefix=SDC4

hw.module.extern @mybuf(in %a: i1 {synth.liberty.pin = {clock = "false"}}, out y: i1)
hw.module.extern @myff(in %clk: i1 {synth.liberty.pin = {clock = "true"}}, in %d: i1, out q: i1)
// Clock gate stub: out = in & (en | test_en).
hw.module @eicg_stub(in %in: i1, in %en: i1, out out: i1) {
  %q = comb.and %in, %en : i1
  hw.output %q : i1
}
// On-chip source stub.
hw.module @fixedclk_stub(out o: i1) {
  %c = hw.constant true
  hw.output %c : i1
}

// SDC2: create_clock -name clk -period 10.000 [get_ports {clk}]
// SDC2: fixedclk_stub/o (1 flops)
hw.module @top2(in %clk: i1, in %d: i1, in %en: i1, out q1: i1, out q2: i1) {
  %g = hw.instance "g" @eicg_stub(in: %clk: i1, en: %en: i1) -> (out: i1)
  %q1 = hw.instance "f1" @myff(clk: %g: i1, d: %d: i1) -> (q: i1)
  %s = hw.instance "s" @fixedclk_stub() -> (o: i1)
  %q2 = hw.instance "f2" @myff(clk: %s: i1, d: %d: i1) -> (q: i1)
  hw.output %q1, %q2 : i1, i1
}

// Shared submodule instantiated under two different clocks: both clocks
// exist (union), one flop each.
// SDC3: Shared-context clock candidates:
// SDC3: c1 (1 flops)
// SDC3: c2 (1 flops)
hw.module @shared(in %clk: i1, in %d: i1, out q: i1) {
  %q = hw.instance "f" @myff(clk: %clk: i1, d: %d: i1) -> (q: i1)
  hw.output %q : i1
}
hw.module @top3(in %c1: i1, in %c2: i1, in %d: i1, out q1: i1, out q2: i1) {
  %q1 = hw.instance "s1" @shared(clk: %c1: i1, d: %d: i1) -> (q: i1)
  %q2 = hw.instance "s2" @shared(clk: %c2: i1, d: %d: i1) -> (q: i1)
  hw.output %q1, %q2 : i1, i1
}

// Sub hierarchy: flop clock traces through a buffer and a port.
hw.module @sub(in %clk: i1, in %d: i1, out q: i1) {
  %b = hw.instance "b" @mybuf(a: %clk: i1) -> (y: i1)
  %q = hw.instance "f" @myff(clk: %b: i1, d: %d: i1) -> (q: i1)
  hw.output %q : i1
}

// Comb-driven clock is reported unconstrained, not constrained.
// CHECK: create_clock -name clk -period 5.000 [get_ports {clk}]
// CHECK: set_input_delay 0.000 -clock [get_clocks {clk}] [get_ports {d e}]
// CHECK: set_output_delay 0.000 -clock [get_clocks {clk}] [get_ports {q r}]
// CHECK: Unconstrained clock at top/g
hw.module @top(in %clk: i1, in %d: i1, in %e: i1, out q: i1, out r: i1) {
  %q = hw.instance "s" @sub(clk: %clk: i1, d: %d: i1) -> (q: i1)
  %gated = comb.and %clk, %e : i1
  %r = hw.instance "g" @myff(clk: %gated: i1, d: %d: i1) -> (q: i1)
  hw.output %q, %r : i1, i1
}

// Uncertainty margins and reset false paths (Phase 1 CTS inputs).
// SDC4: create_clock -name clk -period 5.000 [get_ports {clk}]
// SDC4: set_clock_uncertainty -setup 0.050 [get_clocks {clk}]
// SDC4: set_clock_uncertainty -hold 0.020 [get_clocks {clk}]
// SDC4: set_false_path -from [get_ports {rst}]
hw.module @top4(in %clk: i1, in %d: i1, in %rst: i1, out q: i1) {
  %q = hw.instance "f" @myff(clk: %clk: i1, d: %d: i1) -> (q: i1)
  hw.output %q : i1
}
