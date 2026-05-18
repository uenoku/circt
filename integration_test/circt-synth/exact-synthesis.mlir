// REQUIRES: z3-integration, circt-lec-jit
// RUN: circt-opt %s --lower-comb -o %t.lowered.mlir

// and2 exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,AND2
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// and3 exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:3})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,AND3
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// dot exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3})' -o %t.dot.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,DOT
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// xag exact synthesis.
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:2 allowed-ops=synth.aig.and_inv:2})' -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir --check-prefixes=CHECK,XAG
// RUN: circt-lec %t.lowered.mlir %t.mlir -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// CHECK-LABEL: hw.module @test
// AND2: synth.aig.and_inv {{.*}}, {{.*}} : i1
// AND3: synth.aig.and_inv {{.*}}, {{.*}}, {{.*}} : i1
// DOT: synth.dot {{.*}}, {{.*}}, {{.*}} : i1
// XAG-DAG: synth.xor_inv {{.*}}, {{.*}} : i1
// XAG-DAG: synth.aig.and_inv {{.*}}, {{.*}} : i1
// CHECK-NOT: comb.truth_table
// CHECK: hw.output
// LEC: c1 == c2

hw.module @test(in %a : i1, in %b : i1, in %c : i1,
                                      out and3 : i1, out parity : i1,
                                      out majority : i1) {
  %and3 = comb.truth_table %a, %b, %c -> [false, false, false, false,
                                           false, false, false, true]
  %parity = comb.truth_table %a, %b, %c -> [false, true, true, false,
                                             true, false, false, true]
  %majority = comb.truth_table %a, %b, %c -> [false, false, false, true,
                                               false, true, true, true]
  hw.output %and3, %parity, %majority : i1, i1, i1
}
