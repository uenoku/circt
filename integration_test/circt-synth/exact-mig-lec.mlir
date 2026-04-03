// REQUIRES: z3-integration, libz3
// RUN: circt-synth-dbgen --kind=npn --max-inputs=3 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(hw.module(synth-exact-synthesis{kind=mig sat-solver=z3}))' -o %t.db.mlir
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db-file=%t.db.mlir}))' -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 majority_tree --c2 majority_tree | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 no_change_and --c2 no_change_and | FileCheck %s --check-prefix=LEC

// LEC: c1 == c2

// CHECK-LABEL: hw.module @majority_tree
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[M0:.+]] = synth.mig.maj_inv
// CHECK-NEXT: %[[M1:.+]] = synth.mig.maj_inv
// CHECK-NEXT: %[[M2:.+]] = synth.mig.maj_inv
// CHECK-NOT: synth.aig.and_inv
// CHECK-NEXT: hw.output %[[M2]] : i1
hw.module @majority_tree(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %ab = synth.aig.and_inv %a, %b : i1
  %ac = synth.aig.and_inv %a, %c : i1
  %or0 = synth.aig.and_inv not %ab, not %ac : i1
  %bc = synth.aig.and_inv %b, %c : i1
  %y = synth.aig.and_inv not %or0, not %bc : i1
  hw.output %y : i1
}

// CHECK-LABEL: hw.module @no_change_and
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[M0:.+]] = synth.mig.maj_inv %[[FALSE]], %a, %b : i1
// CHECK-NEXT: hw.output %[[M0]] : i1
hw.module @no_change_and(in %a : i1, in %b : i1, out y : i1) {
  %y = synth.aig.and_inv %a, %b : i1
  hw.output %y : i1
}
