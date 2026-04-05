// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=npn --max-inputs=2 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-greedy-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir max-iterations=1}))' | FileCheck %s

// CHECK-LABEL: hw.module @xor_from_aig
// CHECK: %[[LHS:.+]] = synth.aig.and_inv %a, not %b
// CHECK: %[[RHS:.+]] = synth.aig.and_inv not %a, %b
// CHECK: %[[Y:.+]] = synth.aig.and_inv not %[[LHS]], not %[[RHS]]
// CHECK: hw.output %[[Y]] : i1
hw.module @xor_from_aig(in %a : i1, in %b : i1, out y : i1) {
  %lhs = synth.aig.and_inv %a, not %b : i1
  %rhs = synth.aig.and_inv not %a, %b : i1
  %y = synth.aig.and_inv not %lhs, not %rhs : i1
  hw.output %y : i1
}

// CHECK-LABEL: hw.module @already_optimal
// CHECK: %[[FALSE2:.+]] = hw.constant false
// CHECK: %[[N1:.+]] = synth.mig.maj_inv %[[FALSE2]], %a, not %b
// CHECK: %[[N2:.+]] = synth.mig.maj_inv %[[FALSE2]], not %a, not %b
// CHECK: %[[N3:.+]] = synth.mig.maj_inv not %a, %[[N1]], not %[[N2]]
// CHECK: hw.output %[[N3]] : i1
hw.module @already_optimal(in %a : i1, in %b : i1, out y : i1) {
  %false = hw.constant false
  %0 = synth.mig.maj_inv %false, %a, not %b : i1
  %1 = synth.mig.maj_inv %false, not %a, not %b : i1
  %2 = synth.mig.maj_inv not %a, %0, not %1 : i1
  hw.output %2 : i1
}
