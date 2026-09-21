// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-strip-timing-probes)' | FileCheck %s

// Explicit probe-lifetime terminator: each probe result is rewired to its
// input and the probe erased. Nothing downstream may observe one.

// CHECK-LABEL: hw.module @strip_basic
hw.module @strip_basic(in %a : i1, out z : i1) {
    // CHECK-NOT: synth.timing_probe
    // CHECK: hw.output %a : i1
    %p = synth.timing_probe %a : i1 {required_time = 10000 : i64}
    hw.output %p : i1
}

// Multiple uses of one probe result all see the raw input.
// CHECK-LABEL: hw.module @strip_multi_use
hw.module @strip_multi_use(in %a : i32, out x : i32, out y : i32) {
    // CHECK-NOT: synth.timing_probe
    // CHECK: hw.output %a, %a : i32, i32
    %p = synth.timing_probe %a : i32 {required_time = 5000 : i64}
    hw.output %p, %p : i32, i32
}

// Nested probes collapse to the original driver.
// CHECK-LABEL: hw.module @strip_nested
hw.module @strip_nested(in %a : i1, out z : i1) {
    // CHECK-NOT: synth.timing_probe
    // CHECK: hw.output %a : i1
    %p1 = synth.timing_probe %a : i1 {required_time = 9000 : i64}
    %p2 = synth.timing_probe %p1 : i1 {required_time = 8000 : i64}
    hw.output %p2 : i1
}

// No probes: no-op, module untouched.
// CHECK-LABEL: hw.module @strip_clean
hw.module @strip_clean(in %a : i1, out z : i1) {
    // CHECK-NOT: synth.timing_probe
    // CHECK: hw.output %a : i1
    hw.output %a : i1
}

// Transient refinement records are removed with the probes.
// CHECK-LABEL: hw.module @strip_recorded
// CHECK-NOT: datapath.refine.policy
// CHECK-NOT: synth.timing_probe
hw.module @strip_recorded(in %a : i4, in %b : i4, out z : i4) {
    %p = synth.timing_probe %a : i4 {required_time = 10000 : i64}
    %s = comb.add %p, %b {datapath.refine.policy = "reduce-delay"} : i4
    hw.output %s : i4
}
