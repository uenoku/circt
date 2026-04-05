// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=npn --max-inputs=2 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=dig sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-greedy-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir max-iterations=1}))' | FileCheck %s

// CHECK-LABEL: hw.module @xor_from_aig
// CHECK: %[[LHS:.+]] = synth.aig.and_inv %a, not %b
// CHECK: %[[RHS:.+]] = synth.aig.and_inv not %a, %b
// CHECK: %[[DOT0:.+]] = synth.dig.dot_inv
// CHECK: %[[DOT1:.+]] = synth.dig.dot_inv
// CHECK: hw.output %[[DOT1]] : i1
hw.module @xor_from_aig(in %a : i1, in %b : i1, out y : i1) {
  %lhs = synth.aig.and_inv %a, not %b : i1
  %rhs = synth.aig.and_inv not %a, %b : i1
  %y = synth.aig.and_inv not %lhs, not %rhs : i1
  hw.output %y : i1
}
