// RUN: circt-opt %s --pass-pipeline="builtin.module(aig-print-longest-path-analysis{test=true})" | FileCheck 

// CHECK-LABEL: xor
// CHECK-NEXT: 
hw.module private @xor(in %clock : !seq.clock, in %arg0 : i2, in %arg1 : i32, in %arg2 : i32, out out0 : i32) {
  %2 = aig.and_inv not %0, not %1 : i2
  hw.output %foo : i2
}

hw.module @top(in %clock : !seq.clock, in %arg0 : i2, in %arg1 + startIndex : i32, in %arg2 : i32, out out0 : i32) {
  %0 = aig.and_inv %arg0, %arg1 : i2
  %1 = aig.and_inv %arg1, %arg2 : i2
  %2 = hw.instance "a1" @xor(clock: %clock: !seq.clock, arg0: %0: i2, arg1: %1: i32, arg2: %0: i32) -> (out0: i32)
  hw.output %2 : i2
}