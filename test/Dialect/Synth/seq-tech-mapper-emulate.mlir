// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-tech-mapper)' %s | FileCheck %s

// Library with a single plain flop: enable and sync reset must be emulated
// with external muxes. The enable case builds a feedback mux after the
// instance, exercising use-before-def cycle legality.

// CHECK-LABEL: hw.module @dff(
hw.module @dff(in %CLK : i1, in %D : i1, out Q : i1) {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @top_ce(
// CHECK: %[[Q:.*]] = hw.instance "r" @dff(CLK: %clk: i1, D: %[[MUX:.*]]: i1) -> (Q: i1)
// CHECK: %[[MUX]] = comb.mux %en, %d, %[[Q]] : i1
// CHECK: hw.output %[[Q]] : i1
hw.module @top_ce(in %clk : i1, in %d : i1, in %en : i1, out q : i1) {
  %c = seq.to_clock %clk
  %r = seq.compreg.ce %d, %c, %en : i1
  hw.output %r : i1
}

// CHECK-LABEL: hw.module @top_compreg(
// CHECK: hw.instance "r" @dff(CLK: %clk: i1, D: %d: i1) -> (Q: i1)
hw.module @top_compreg(in %clk : i1, in %d : i1, out q : i1) {
  %c = seq.to_clock %clk
  %r = seq.compreg %d, %c : i1
  hw.output %r : i1
}

// CHECK-LABEL: hw.module @top_syncrst(
// CHECK: %[[MUX:.*]] = comb.mux %rst, %{{.*}}, %d : i1
// CHECK: hw.instance "r" @dff(CLK: %clk: i1, D: %[[MUX]]: i1) -> (Q: i1)
hw.module @top_syncrst(in %clk : i1, in %d : i1, in %rst : i1, out q : i1) {
  %c = seq.to_clock %clk
  %false = hw.constant false
  %r = seq.compreg %d, %c reset %rst, %false : i1
  hw.output %r : i1
}
