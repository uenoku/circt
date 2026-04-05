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
// CHECK-NEXT: %[[TRUE:.+]] = hw.constant true
// CHECK-NEXT: %[[ROOT:.+]] = synth.dig.dot_inv %a, %[[TRUE]], %c : i1
// CHECK-NEXT: %[[KEEP:.+]] = comb.xor %a, %[[FALSE]] : i1
// CHECK-NEXT: hw.output %[[ROOT]], %[[KEEP]] : i1, i1
hw.module @keep_shared_y(in %a : i1, in %c : i1, out y : i1, out keep : i1) {
  %false = hw.constant false
  %inner = synth.dig.dot_inv %a, %false, %false : i1
  %root = synth.dig.dot_inv %a, %inner, %c : i1
  %keep = comb.xor %inner, %false : i1
  hw.output %root, %keep : i1, i1
}

// CHECK-LABEL: hw.module @keep_y_when_x_reuses_it
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[FALSE2:.+]] = hw.constant false
// CHECK-NEXT: %[[ROOT:.+]] = synth.dig.dot_inv not %a, %[[FALSE2]], %c : i1
// CHECK-NEXT: hw.output %[[ROOT]] : i1
hw.module @keep_y_when_x_reuses_it(in %a : i1, in %c : i1, out y : i1) {
  %false = hw.constant false
  %inner = synth.dig.dot_inv %a, %false, %false : i1
  %root = synth.dig.dot_inv not %inner, %inner, %c : i1
  hw.output %root : i1
}

// CHECK-LABEL: hw.module @rewrite_shared_y_when_root_folds
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[FALSE2:.+]] = hw.constant false
// CHECK-NEXT: %[[KEEP:.+]] = comb.xor %a, %[[FALSE]] : i1
// CHECK-NEXT: hw.output %[[FALSE2]], %[[KEEP]] : i1, i1
hw.module @rewrite_shared_y_when_root_folds(in %a : i1, out y : i1, out keep : i1) {
  %false = hw.constant false
  %inner = synth.dig.dot_inv %a, %false, %false : i1
  %root = synth.dig.dot_inv %a, %inner, %false : i1
  %keep = comb.xor %inner, %false : i1
  hw.output %root, %keep : i1, i1
}
