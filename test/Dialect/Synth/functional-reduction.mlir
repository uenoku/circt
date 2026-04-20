// RUN: circt-opt %s --synth-functional-reduction=test-transformation | FileCheck %s

// AND(AND(a, not b), AND(c, not d)) is equivalent to AND(a, not b, c, not d).
// CHECK-LABEL: hw.module @test_mixed
hw.module @test_mixed(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: %[[RESULT_0:.+]] = synth.aig.and_inv %a, not %b
  // CHECK: %[[RESULT_1:.+]] = synth.aig.and_inv %c, not %d
  // CHECK: %[[RESULT_2:.+]] = synth.aig.and_inv %[[RESULT_0]], %[[RESULT_1]]
  // CHECK: %[[RESULT_3:.+]] = synth.aig.and_inv %a, not %b, %c, not %d
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[RESULT_2]], %[[RESULT_3]]
  // CHECK: %[[RESULT_4:.+]] = synth.aig.and_inv %a, not %b, not %c, not %d
  // CHECK-NEXT: hw.output %[[CHOICE]], %[[CHOICE]], %[[RESULT_4]]
  %0 = synth.aig.and_inv %a, not %b : i1
  %1 = synth.aig.and_inv %c, not %d : i1
  %2 = synth.aig.and_inv %0, %1 {synth.test.fc_equiv_class = 0} : i1
  %3 = synth.aig.and_inv %a, not %b, %c, not %d {synth.test.fc_equiv_class = 0} : i1
  %4 = synth.aig.and_inv %a, not %b, not %c, not %d {synth.test.fc_equiv_class = 1} : i1
  hw.output %2, %3, %4 : i1, i1, i1
}

// CHECK-LABEL: hw.module @test_supported_ops
hw.module @test_supported_ops(in %a: i1, in %b: i1, in %c: i1,
                              out out0: i1, out out1: i1,
                              out out2: i1, out out3: i1,
                              out out4: i1, out out5: i1) {
  // CHECK: %[[OR0:.+]] = comb.or %a, %b, %c
  // CHECK: %[[OR1:.+]] = comb.or %c, %b, %a
  // CHECK-NEXT: %[[ORCHOICE:.+]] = synth.choice %[[OR0]], %[[OR1]] : i1
  // CHECK: %[[XOR0:.+]] = comb.xor %a, %b
  // CHECK: %[[XOR1:.+]] = comb.xor %b, %a
  // CHECK-NEXT: %[[XORCHOICE:.+]] = synth.choice %[[XOR0]], %[[XOR1]] : i1
  // CHECK: %[[AND0:.+]] = comb.and %a, %b
  // CHECK: %[[AND1:.+]] = comb.and %b, %a
  // CHECK-NEXT: %[[ANDCHOICE:.+]] = synth.choice %[[AND0]], %[[AND1]] : i1
  // CHECK: hw.output %[[ORCHOICE]], %[[ORCHOICE]], %[[XORCHOICE]], %[[XORCHOICE]], %[[ANDCHOICE]], %[[ANDCHOICE]] : i1, i1, i1, i1, i1, i1
  %0 = comb.or %a, %b, %c {synth.test.fc_equiv_class = 2} : i1
  %1 = comb.or %c, %b, %a {synth.test.fc_equiv_class = 2} : i1
  %2 = comb.xor %a, %b {synth.test.fc_equiv_class = 4} : i1
  %3 = comb.xor %b, %a {synth.test.fc_equiv_class = 4} : i1
  %4 = comb.and %a, %b {synth.test.fc_equiv_class = 5} : i1
  %5 = comb.and %b, %a {synth.test.fc_equiv_class = 5} : i1
  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @test_inversion_equiv
hw.module @test_inversion_equiv(in %a: i1, in %b: i1, out out0: i1, out out1: i1) {
  // CHECK: %[[AND:.+]] = synth.aig.and_inv not %a, not %b
  // CHECK: %[[OR:.+]] = comb.or %a, %b
  // CHECK: %[[NOTMEMBER:.+]] = synth.aig.and_inv not %[[OR]]
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[AND]], %[[NOTMEMBER]] : i1
  // CHECK: %[[CHOICENOT:.+]] = synth.aig.and_inv not %[[CHOICE]]
  // CHECK: hw.output %[[CHOICE]], %[[CHOICENOT]]
  %0 = synth.aig.and_inv not %a, not %b {synth.test.fc_equiv_class = 10} : i1
  %1 = comb.or %a, %b {synth.test.fc_equiv_class = 10} : i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: hw.module @test_no_ssa_cycle
hw.module @test_no_ssa_cycle(in %a: i1, in %b: i1,
                             out out0: i1, out out1: i1, out out2: i1) {
  // CHECK: %[[AND0:.+]] = synth.aig.and_inv %a, %b
  // CHECK: %[[AND1:.+]] = synth.aig.and_inv %[[AND0]], %a
  // CHECK: %[[AND2:.+]] = synth.aig.and_inv %b, %a
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[AND0]], %[[AND1]], %[[AND2]]
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]], %[[CHOICE]]
  %0 = synth.aig.and_inv %a, %b {synth.test.fc_equiv_class = 7} : i1
  %1 = synth.aig.and_inv %0, %a {synth.test.fc_equiv_class = 7} : i1
  %2 = synth.aig.and_inv %b, %a {synth.test.fc_equiv_class = 7} : i1
  hw.output %0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: hw.module @test_xor_inv_equiv
hw.module @test_xor_inv_equiv(in %a: i1, in %b: i1, out out0: i1, out out1: i1) {
  // CHECK: %[[XOR_INV:.+]] = synth.xor_inv %a, %b
  // CHECK: %[[XOR:.+]] = comb.xor %b, %a
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[XOR_INV]], %[[XOR]] : i1
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]] : i1, i1
  %0 = synth.xor_inv %a, %b {synth.test.fc_equiv_class = 11} : i1
  %1 = comb.xor %b, %a {synth.test.fc_equiv_class = 11} : i1
  hw.output %0, %1 : i1, i1
}

// CHECK-LABEL: hw.module @test_xor_inv_input_inversion
hw.module @test_xor_inv_input_inversion(in %a: i1, in %b: i1,
                                        out out0: i1, out out1: i1) {
  // CHECK: %[[XOR_INV:.+]] = synth.xor_inv not %a, %b
  // CHECK: %[[XOR:.+]] = comb.xor %a, %b
  // CHECK: %[[NOT_XOR:.+]] = synth.aig.and_inv not %[[XOR]]
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[XOR_INV]], %[[NOT_XOR]] : i1
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]] : i1, i1
  %0 = synth.xor_inv not %a, %b {synth.test.fc_equiv_class = 12} : i1
  %1 = comb.xor %a, %b : i1
  %2 = synth.aig.and_inv not %1 {synth.test.fc_equiv_class = 12} : i1
  hw.output %0, %2 : i1, i1
}

// CHECK-LABEL: hw.module @test_dot_input_inversion
hw.module @test_dot_input_inversion(in %a: i1, in %b: i1, in %c: i1,
                                    out out0: i1, out out1: i1) {
  // CHECK: %[[NOT_A:.+]] = synth.aig.and_inv not %a
  // CHECK: %[[NOT_C:.+]] = synth.aig.and_inv not %c
  // CHECK: %[[AND:.+]] = comb.and %[[NOT_A]], %b
  // CHECK: %[[OR:.+]] = comb.or %[[NOT_C]], %[[AND]]
  // CHECK: %[[XOR:.+]] = comb.xor %[[NOT_A]], %[[OR]]
  // CHECK: %[[DOT:.+]] = synth.dot not %a, %b, not %c
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[XOR]], %[[DOT]] : i1
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]] : i1, i1
  %notA = synth.aig.and_inv not %a : i1
  %notC = synth.aig.and_inv not %c : i1
  %0 = comb.and %notA, %b : i1
  %1 = comb.or %notC, %0 : i1
  %2 = comb.xor %notA, %1 {synth.test.fc_equiv_class = 13} : i1
  %3 = synth.dot not %a, %b, not %c {synth.test.fc_equiv_class = 13} : i1
  hw.output %2, %3 : i1, i1
}
