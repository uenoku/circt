// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse, datapath-reduce-delay, convert-datapath-to-comb, convert-comb-to-synth, cse, canonicalize))' | FileCheck %s

// Phase-1 spanning proof at pass level: a hand-placed probe rides the full
// pre-map datapath pipeline (Comb -> Datapath, canonicalize, CSE,
// datapath-reduce-delay, Datapath -> Comb, Comb -> Synth) as a transparent
// pass-through with its required_time intact.

// CHECK-LABEL: hw.module @arith
hw.module @arith(in %a : i8, in %b : i8, in %c : i8, out z : i8) {
  %add = comb.add %a, %b, %c : i8
  %mul = comb.mul %a, %b : i8
  %sum = comb.add %add, %mul : i8
  %p = synth.timing_probe %sum : i8 {required_time = 10000 : i64}
  hw.output %p : i8
}

// The arithmetic is lowered (no datapath ops remain) and exactly one probe
// with the original requirement survives on the lowered sum.
// CHECK-NOT: datapath.
// CHECK-COUNT-1: synth.timing_probe {{.*}} {required_time = 10000 : i64}
// CHECK-NOT: datapath.
