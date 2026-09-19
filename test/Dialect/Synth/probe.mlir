// RUN: circt-opt %s | FileCheck %s

// Transparent pass-through with a required time.
// CHECK-LABEL: hw.module @probe_ok
hw.module @probe_ok(in %a : i1, out z : i1) {
    // CHECK: %[[P:.+]] = synth.timing_probe %a : i1 {required_time = 10000 : i64}
    %p = synth.timing_probe %a : i1 {required_time = 10000 : i64}
    // CHECK: hw.output %[[P]] : i1
    hw.output %p : i1
}
