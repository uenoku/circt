// RUN: circt-opt --pass-pipeline='builtin.module(synth-strip-lib-to-extern)' %s | FileCheck %s

// Library cells collapse to declarations; design modules are untouched.

// CHECK: hw.module.extern @and2(in %a : i1, in %b : i1, out x : i1)
// CHECK-NOT: comb.and
// CHECK-LABEL: hw.module @top(
// CHECK: hw.instance "i" @and2(a: %a: i1, b: %b: i1) -> (x: i1)
hw.module @and2(in %a : i1, in %b : i1, out x : i1) attributes {synth.liberty.cell = {area = 1.0 : f64}} {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}
hw.module @top(in %a : i1, in %b : i1, out o : i1) {
  %0 = hw.instance "i" @and2(a: %a: i1, b: %b: i1) -> (x: i1)
  hw.output %0 : i1
}
