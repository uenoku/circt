// REQUIRES: z3-integration, libz3
// REQUIRES: circt-lec-jit

// RUN: circt-synth %s --target-ir=mig --until-before mapping -o %t.orig.mlir
// RUN: circt-synth-dbgen --kind=npn --max-inputs=3 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig-xor sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %t.orig.mlir -pass-pipeline='builtin.module(hw.module(synth-greedy-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir max-iterations=1}))' -o %t.greedy.mlir
// RUN: circt-lec %t.orig.mlir %t.greedy.mlir --shared-libs=%libz3 --c1 add16 --c2 add16 | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %t.orig.mlir %t.greedy.mlir --shared-libs=%libz3 --c1 add33 --c2 add33 | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %t.orig.mlir %t.greedy.mlir --shared-libs=%libz3 --c1 add65 --c2 add65 | FileCheck %s --check-prefix=LEC

// LEC: c1 == c2
hw.module @add16(in %arg0: i16, in %arg1: i16, in %arg2: i16, out add: i16) {
  %0 = comb.add %arg0, %arg1, %arg2 : i16
  hw.output %0 : i16
}

hw.module @add33(in %arg0: i33, in %arg1: i33, in %arg2: i33, out add: i33) {
  %0 = comb.add %arg0, %arg1, %arg2 : i33
  hw.output %0 : i33
}

hw.module @add65(in %arg0: i65, in %arg1: i65, in %arg2: i65, out add: i65) {
  %0 = comb.add %arg0, %arg1, %arg2 : i65
  hw.output %0 : i65
}
