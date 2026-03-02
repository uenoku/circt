// RUN: circt-opt --synth-gen-supergates %s | FileCheck %s
// RUN: circt-opt --synth-gen-supergates="max-gates=3" %s -split-input-file > /dev/null

// NOR2 cell: out = ~(a | b) = ~a & ~b
hw.module @NOR2(in %a : i1, in %b : i1, out Y : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[100], [100]]}} {
    %0 = synth.aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

// INV cell: out = ~a
hw.module @INV(in %a : i1, out Y : i1) attributes {hw.techlib.info = {area = 0.5 : f64, delay = [[50]]}} {
    %0 = synth.aig.and_inv not %a : i1
    hw.output %0 : i1
}

// CHECK: hw.module @NOR2
// CHECK: hw.module @INV

// NOR2(NOR2(a,b), c) = (a|b) & ~c — a 3-input function not covered by NOR2/INV.
// CHECK:      hw.module private @__supergate_
// CHECK-SAME: hw.techlib.info = {area = 2.000000e+00 : f64, delay =
// CHECK-SAME: synth.supergate = true
// CHECK:      %[[INNER:.*]] = hw.instance "inner" @NOR2
// CHECK:      %[[OUTER:.*]] = hw.instance "outer" @NOR2

// -----

// RUN: circt-opt --synth-gen-supergates="max-inputs=2" %s -split-input-file | FileCheck %s --check-prefix=MAXIN

// AND2 cell: out = a & b
hw.module @AND2(in %a : i1, in %b : i1, out Y : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[100], [100]]}} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// INV2 cell: out = ~a
hw.module @INV2(in %a : i1, out Y : i1) attributes {hw.techlib.info = {area = 0.5 : f64, delay = [[50]]}} {
    %0 = synth.aig.and_inv not %a : i1
    hw.output %0 : i1
}

// With max-inputs=2, only 2-input supergates are allowed.
// All depth-2 compositions here are NPN-equivalent to covered primitive classes.
// MAXIN: hw.module @AND2
// MAXIN: hw.module @INV2
// MAXIN-NOT: hw.module private @__supergate_

// -----

// RUN: circt-opt --synth-gen-supergates="max-gates=3 max-inputs=6" %s -split-input-file | FileCheck %s --check-prefix=FOURIN

hw.module @AND4_SRC(in %a : i1, in %b : i1, out Y : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[100], [100]]}} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// FOURIN: hw.module @AND4_SRC
// FOURIN: hw.module private @__supergate_
// FOURIN: in %{{.*}} : i1, in %{{.*}} : i1, in %{{.*}} : i1, in %{{.*}} : i1, out Y : i1
// FOURIN: synth.supergate = true
