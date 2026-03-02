// RUN: circt-synth %s --top top | FileCheck %s

hw.module @AND3(in %a : i1, in %b : i1, in %c : i1, out Y : i1) attributes {hw.techlib.info = {area = 2.0 : f64, delay = [[200], [200], [100]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  hw.output %1 : i1
}

hw.module @top(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, out Y : i1) {
  hw.output %a : i1
}

// CHECK: hw.module private @__supergate_
// CHECK-SAME: synth.supergate = true
// CHECK: %{{.*}} = hw.instance "inner" @AND3
// CHECK: %{{.*}} = hw.instance "outer" @AND3
