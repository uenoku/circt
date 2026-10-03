// RUN: circt-synth %s --rover --rover-patterns=%S/rover-patterns.mlir --rover-iters=2 | FileCheck %s

// Rover runs before lowering: mul+shl datapath idiom saturates and extracts
// without leaving e-graph ops behind, then synthesizes to bits.
// CHECK-LABEL: hw.module @ShiftMult
// CHECK-NOT: equivalence.graph
// CHECK-NOT: equivalence.class
// CHECK: hw.output
hw.module @ShiftMult(in %a : i32, in %b : i32, in %m : i5, in %n : i5, out result : i64) {
  %c0_i59 = hw.constant 0 : i59
  %c0_i32 = hw.constant 0 : i32
  %0 = comb.concat %c0_i32, %a : i32, i32
  %1 = comb.concat %c0_i59, %m : i59, i5
  %2 = comb.shl %0, %1 : i64
  %3 = comb.concat %c0_i32, %b : i32, i32
  %4 = comb.concat %c0_i59, %n : i59, i5
  %5 = comb.shl %3, %4 : i64
  %6 = comb.mul %2, %5 : i64
  hw.output %6 : i64
}
