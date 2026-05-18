// REQUIRES: z3-integration, circt-lec-jit
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2})' -o %t
// RUN: FileCheck %s --input-file=%t.and2.mlir --check-prefixes=CHECK,AND2
// RUN: circt-lec %s %t -c1=test -c2=test --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:3})' -o %t.and3.mlir
// RUN: FileCheck %s --input-file=%t.and3.mlir --check-prefix=IR
// RUN: circt-lec %s %t.and3.mlir -c1=three_input_functions_ref -c2=three_input_functions_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.majority:3})' -o %t.maj3.mlir
// RUN: FileCheck %s --input-file=%t.maj3.mlir --check-prefix=IR
// RUN: circt-lec %s %t.maj3.mlir -c1=three_input_functions_ref -c2=three_input_functions_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3})' -o %t.dot.mlir
// RUN: FileCheck %s --input-file=%t.dot.mlir --check-prefix=IR
// RUN: circt-lec %s %t.dot.mlir -c1=three_input_functions_ref -c2=three_input_functions_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:2 allowed-ops=synth.aig.and_inv:2})' -o %t.xor-and.mlir
// RUN: FileCheck %s --input-file=%t.xor-and.mlir --check-prefix=IR
// RUN: circt-lec %s %t.xor-and.mlir -c1=three_input_functions_ref -c2=three_input_functions_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.majority:3 allowed-ops=synth.xor_inv:2})' -o %t.maj-xor.mlir
// RUN: FileCheck %s --input-file=%t.maj-xor.mlir --check-prefix=IR
// RUN: circt-lec %s %t.maj-xor.mlir -c1=three_input_functions_ref -c2=three_input_functions_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=LEC

// CHECK-LABEL: hw.module @three_input_functions_impl
// CHECK-NOT: comb.truth_table
// LEC: c1 == c2

// AND2: synth.aig.and_inv {{.*}}, {{.*}} : i1
// AND3: synth.aig.and_inv {{.*}}, {{.*}}, {{.*}} : i1

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
