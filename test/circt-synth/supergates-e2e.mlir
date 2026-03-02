// RUN: circt-synth %s --top test | FileCheck %s
// XFAIL: *

hw.module @AND2(in %a : i1, in %b : i1, out Y : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[100], [100]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @OR2(in %a : i1, in %b : i1, out Y : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[100], [100]]}} {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

hw.module @test(in %a : i1, in %b : i1, out Y : i1) {
  %ac = synth.aig.and_inv not %a, %b : i1
  hw.output %ac : i1
}

// CHECK: hw.module private @__supergate_
// CHECK-SAME: synth.supergate = true
// CHECK-LABEL: hw.module @test
// CHECK: %[[SG:.+]] = hw.instance "mapped" @__supergate_
// CHECK-NOT: hw.instance "mapped" @AND2
// CHECK-NOT: hw.instance "mapped" @OR2
// CHECK: hw.output %[[SG]] : i1
