// RUN: circt-synth --liberty %S/Inputs/probe.lib --probe-clocks=clk --buffer-high-fanout --max-fanout=2 %s | FileCheck %s
// RUN: circt-synth --liberty %S/Inputs/probe.lib --probe-clocks=clk %s | FileCheck %s --check-prefix=STRIP

// Phase-1 probe lifecycle: probes are derived early on mapped flop D inputs
// (all five flops here), ride the lowering/mapping pipeline as transparent
// pass-throughs, and never leak downstream. With buffering on, the buffer
// pass consumes them after splitting the r1.Q broadcast (fanout 4 into two
// chunks behind two buffers). With buffering off, the explicit strip
// terminator removes them.

// CHECK-NOT: synth.timing_probe
// CHECK-LABEL: hw.module @top
// CHECK: hw.instance "r1" @mydff(D: %a: i1, CLK: %clk: i1)
// CHECK-COUNT-2: hw.instance "buffered_{{.+}}" @mybuf(A: %r1.Q: i1)
// CHECK: hw.instance "r2" @mydff
// CHECK-NOT: synth.timing_probe

// STRIP-NOT: synth.timing_probe
// STRIP-LABEL: hw.module @top
// STRIP-NOT: @mybuf
// STRIP: hw.instance "r1" @mydff(D: %a: i1, CLK: %clk: i1)
// STRIP: hw.instance "r2" @mydff
// STRIP-NOT: synth.timing_probe
hw.module @top(in %clk : i1, in %a : i1, in %b0 : i1, in %b1 : i1, in %b2 : i1, in %b3 : i1, out q0 : i1, out q1 : i1, out q2 : i1, out q3 : i1) {
  %c = seq.to_clock %clk
  %r1 = seq.firreg %a clock %c : i1
  %g0 = comb.and %r1, %b0 : i1
  %g1 = comb.and %r1, %b1 : i1
  %g2 = comb.and %r1, %b2 : i1
  %g3 = comb.and %r1, %b3 : i1
  %r2 = seq.firreg %g0 clock %c : i1
  %r3 = seq.firreg %g1 clock %c : i1
  %r4 = seq.firreg %g2 clock %c : i1
  %r5 = seq.firreg %g3 clock %c : i1
  hw.output %r2, %r3, %r4, %r5 : i1, i1, i1, i1
}
