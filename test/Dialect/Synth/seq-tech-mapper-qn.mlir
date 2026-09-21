// RUN: circt-opt --pass-pipeline='builtin.module(synth-seq-tech-mapper)' %s | FileCheck %s

// Library cells (synth.ff bodies, as produced by the Liberty importer).
// dffQN models a QN-only cell: D inverted in the body, stored value exposed.
hw.module @dff(in %CLK : i1, in %D : i1, out Q : i1) attributes {synth.liberty.cell = {area = 20.0 : f64}} {
  %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
  hw.output %q : i1
}

// CHECK-LABEL: hw.module @dffQN(
// Models a QN-only cell exactly as the Liberty importer builds it: D
// inverted (xor-with-true) into the flop, stored value exposed.
hw.module @dffQN(in %CLK : i1, in %D : i1, out QN : i1) attributes {synth.liberty.cell = {area = 15.0 : f64}} {
  %true = hw.constant true
  %n = comb.xor %D, %true : i1
  %q, %qn = synth.ff %n clock %CLK edge rise : i1, i1
  hw.output %q : i1
}

// The cheaper QN cell is selected, and the inverted-at-port D input is
// compensated by inverting the driver: QN exposes NOT(stored
// NOT(driver)) = driver, preserving the source flop output.
// CHECK-LABEL: hw.module @top_qn(
// CHECK: %[[C:.+]] = hw.constant true
// CHECK: %[[ND:.+]] = comb.xor %d, %[[C]] : i1
// CHECK: hw.instance "r" @dffQN(CLK: %clk: i1, D: %[[ND]]: i1) -> (QN: i1)
hw.module @top_qn(in %clk : i1, in %d : i1, out q : i1) {
  %c = seq.to_clock %clk
  %r = seq.firreg %d clock %c : i1
  hw.output %r : i1
}
