// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-timing-v2-guided-rewrite{print-report}))' | FileCheck %s

module {
  hw.module @TimingV2Guided(in %c0 : i1, in %c1 : i1, in %c2 : i1, in %c3 : i1,
                            in %v0 : i8, in %v1 : i8, in %v2 : i8, in %v3 : i8,
                            in %d : i8, out y : i8) {
    %m3 = comb.mux bin %c3, %v3, %d : i8
    %m2 = comb.mux bin %c2, %v2, %m3 : i8
    %m1 = comb.mux bin %c1, %v1, %m2 : i8
    %m0 = comb.mux bin %c0, %v0, %m1 : i8
    hw.output %m0 : i8
  }
}

// CHECK: TimingV2 guided rewrite report
// CHECK: module: @TimingV2Guided
// CHECK: initial_delay: 4
// CHECK: pattern: balance false-side priority mux chain
// CHECK: rewrite: priority_mux_chain conditions=4 before=4 predicted=3 after=3
// CHECK: rewrites_applied: 1
// CHECK: final_delay: 3

// CHECK-LABEL: hw.module @TimingV2Guided
// CHECK-DAG: comb.or bin %c0, %c1 : i1
// CHECK-NOT: comb.mux bin %c1
// CHECK-NOT: comb.mux bin %c2
// CHECK: hw.output
