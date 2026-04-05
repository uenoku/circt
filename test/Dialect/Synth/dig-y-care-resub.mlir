// REQUIRES: z3-integration
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-dig-y-care-resub{sat-solver=z3 num-random-patterns=64}))' | FileCheck %s

// CHECK-LABEL: hw.module @rewrite_y_to_const
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[TRUE:.+]] = hw.constant true
// CHECK-NEXT: %[[ROOT:.+]] = synth.dig.dot_inv %a, %[[TRUE]], %c : i1
// CHECK-NEXT: hw.output %[[ROOT]] : i1
hw.module @rewrite_y_to_const(in %a : i1, in %c : i1, out y : i1) {
  %false = hw.constant false
  %inner = synth.dig.dot_inv %a, %false, %false : i1
  %root = synth.dig.dot_inv %a, %inner, %c : i1
  hw.output %root : i1
}

// CHECK-LABEL: hw.module @keep_shared_y
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[INNER:.+]] = synth.dig.dot_inv %a, %[[FALSE]], %[[FALSE]] : i1
// CHECK-NEXT: %[[ROOT:.+]] = synth.dig.dot_inv %a, %[[INNER]], %c : i1
// CHECK-NEXT: %[[KEEP:.+]] = comb.xor %[[INNER]], %[[FALSE]] : i1
// CHECK-NEXT: hw.output %[[ROOT]], %[[KEEP]] : i1, i1
hw.module @keep_shared_y(in %a : i1, in %c : i1, out y : i1, out keep : i1) {
  %false = hw.constant false
  %inner = synth.dig.dot_inv %a, %false, %false : i1
  %root = synth.dig.dot_inv %a, %inner, %c : i1
  %keep = comb.xor %inner, %false : i1
  hw.output %root, %keep : i1, i1
}
