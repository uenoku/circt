// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(llhd-strip-init-drives,llhd-sig2reg))' %s | FileCheck %s

// A constant drive next to a real driver is power-up only and goes away,
// letting Sig2Reg fold the signal to the flop output.

// CHECK-LABEL: hw.module @init_plus_driver(
// CHECK-NOT: llhd.sig
// CHECK-NOT: llhd.drv
// CHECK: seq.firreg
hw.module @init_plus_driver(in %clk : i1, in %rst : i1, in %d : i8, out q : i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %false = hw.constant false
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  llhd.drv %sig, %c0_i8 after %0 : i8
  %c = seq.to_clock %clk
  %r = seq.firreg %d clock %c reset async %rst, %c0_i8 : i8
  llhd.drv %sig, %r after %0 : i8
  %v = llhd.prb %sig : i8
  hw.output %v : i8
}

// Differing constants are a genuine conflict: the signal survives.
// CHECK-LABEL: hw.module @conflict(
// CHECK: llhd.sig
hw.module @conflict(in %clk : i1, out q : i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %sig = llhd.sig %c0_i8 : i8
  llhd.drv %sig, %c0_i8 after %0 : i8
  llhd.drv %sig, %c1_i8 after %0 : i8
  %v = llhd.prb %sig : i8
  hw.output %v : i8
}

// A lone constant drive is a tie, not init: stripping leaves it alone
// and Sig2Reg folds it away entirely.
// CHECK-LABEL: hw.module @lone_init(
// CHECK-NOT: llhd.sig
// CHECK-NOT: llhd.drv
hw.module @lone_init(in %clk : i1, out q : i8) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i8 = hw.constant 0 : i8
  %sig = llhd.sig %c0_i8 : i8
  llhd.drv %sig, %c0_i8 after %0 : i8
  %v = llhd.prb %sig : i8
  hw.output %v : i8
}
