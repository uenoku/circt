// RUN: circt-opt --synth-tie-cells %s | FileCheck %s

// Fake liberty tie cell: no inputs, constant-function outputs.
// CHECK-LABEL: hw.module @conb_like
hw.module @conb_like(out HI: i1 {synth.liberty.pin = {function = "1"}}, out LO: i1 {synth.liberty.pin = {function = "0"}}) attributes {synth.liberty.cell = {area = 1.0 : f64}} {
  %t = hw.constant true
  %f = hw.constant false
  hw.output %t, %f : i1, i1
}

hw.module.extern @inv(in %a: i1, out y: i1)

// CHECK-LABEL: hw.module @top
hw.module @top(in %a: i1, out x: i1, out y: i1) {
  %c1 = hw.constant true
  %c0 = hw.constant false
  // A shared tie instance is created per polarity at the block start.
  // CHECK-DAG: hw.instance "tied_hi" @conb_like() -> (HI: i1, LO: i1)
  // CHECK-DAG: hw.instance "tied_lo" @conb_like() -> (HI: i1, LO: i1)
  // CHECK: hw.instance "m1" @inv(a: %tied_hi.HI: i1)
  // CHECK: hw.instance "m2" @inv(a: %tied_lo.LO: i1)
  %y1 = hw.instance "m1" @inv(a: %c1: i1) -> (y: i1)
  %y2 = hw.instance "m2" @inv(a: %c0: i1) -> (y: i1)
  // CHECK: hw.output
  hw.output %y1, %y2 : i1, i1
  // CHECK-NOT: hw.constant
}
