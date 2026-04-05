// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=npn --max-inputs=2 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=dig sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir strategy=area}))' | FileCheck %s

// CHECK-LABEL: hw.module @xor_from_aig
// CHECK-NOT: synth.aig.and_inv
// CHECK: %[[DOT:.+]] = synth.dig.dot_inv
// CHECK: %[[INV:.+]] = synth.dig.dot_inv not %[[DOT]]
// CHECK: hw.output %[[INV]] : i1
hw.module @xor_from_aig(in %a : i1, in %b : i1, out y : i1) {
  %lhs = synth.aig.and_inv %a, not %b : i1
  %rhs = synth.aig.and_inv not %a, %b : i1
  %y = synth.aig.and_inv not %lhs, not %rhs : i1
  hw.output %y : i1
}
