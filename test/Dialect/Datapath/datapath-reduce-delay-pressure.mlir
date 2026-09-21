// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(datapath-reduce-delay))' | FileCheck %s --check-prefix=LEGACY
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(datapath-reduce-delay{pressure-margin-levels=0}))' | FileCheck %s

// Selective rewriting: two identical adder chains, one feeding a probe and
// one relaxed. Legacy mode folds both into compressors; pressure-gated mode
// folds only the probe-fed chain and leaves the relaxed chain
// byte-for-byte structurally unchanged.
// LEGACY-COUNT-2: datapath.compress
// CHECK: %[[COMP:.+]]:2 = datapath.compress %a, %b, %c : i4 [3 -> 2]
// CHECK: %[[SUM:.+]] = comb.add bin %[[COMP]]#0, %[[COMP]]#1 : i4
// CHECK: %[[P:.+]] = synth.timing_probe %[[SUM]] : i4 {required_time = 10000 : i64}
// CHECK: %[[T:.+]] = comb.add %d, %e : i4
// CHECK: %[[S:.+]] = comb.add %[[T]], %f : i4
// CHECK: hw.output %[[P]], %[[S]] : i4, i4
hw.module @two(in %a : i4, in %b : i4, in %c : i4, in %d : i4, in %e : i4, in %f : i4, out x : i4, out y : i4) {
  %t0 = comb.add %a, %b : i4
  %s0 = comb.add %t0, %c : i4
  %p = synth.timing_probe %s0 : i4 {required_time = 10000 : i64}
  %t1 = comb.add %d, %e : i4
  %s1 = comb.add %t1, %f : i4
  hw.output %p, %s1 : i4, i4
}
