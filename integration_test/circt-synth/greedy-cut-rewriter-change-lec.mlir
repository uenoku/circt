// REQUIRES: z3-integration, libz3
// REQUIRES: circt-lec-jit

// RUN: circt-synth %s --target-ir=aig --until-before mapping -o %t.orig.mlir
// RUN: circt-synth-dbgen --kind=npn --max-inputs=3 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig-xor sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %t.orig.mlir -pass-pipeline='builtin.module(hw.module(synth-greedy-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir max-iterations=1}))' -o %t.greedy.mlir
// RUN: not cmp -s %t.orig.mlir %t.greedy.mlir
// RUN: circt-lec %t.orig.mlir %t.greedy.mlir --shared-libs=%libz3 --c1 xor_bank32 --c2 xor_bank32 | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %t.orig.mlir %t.greedy.mlir --shared-libs=%libz3 --c1 xor_bank65 --c2 xor_bank65 | FileCheck %s --check-prefix=LEC

// LEC: c1 == c2
hw.module @xor_bank32(in %a: i32, in %b: i32, out y: i32) {
  %0 = comb.xor %a, %b : i32
  hw.output %0 : i32
}

hw.module @xor_bank65(in %a: i65, in %b: i65, out y: i65) {
  %0 = comb.xor %a, %b : i65
  hw.output %0 : i65
}
