// RUN: circt-opt %s --synth-functional-reduction | FileCheck %s

// Test basic equivalence detection with AndInverterOp
// CHECK-LABEL: hw.module @test_and_equiv
hw.module @test_and_equiv(in %a: i1, in %b: i1, out out1: i1, out out2: i1) {
  // Two equivalent AND operations with same inputs (different order)
  // These should be detected as equivalent by simulation and proven by SAT
  // CHECK: %[[AND:.+]] = synth.aig.and_inv %a, %b : i1
  // CHECK: hw.output %[[AND]], %[[AND]]
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %b, %a : i1
  hw.output %0, %1 : i1, i1
}

// Test complement detection (one is NOT of the other)
// CHECK-LABEL: hw.module @test_complement
hw.module @test_complement(in %a: i1, in %b: i1, out out1: i1, out out2: i1) {
  // %0 = a AND b
  // The last operation computes NOT(a AND b) via De Morgan
  // CHECK: %[[AND:.+]] = synth.aig.and_inv %a, %b : i1
  // CHECK: %[[NOT:.+]] = synth.aig.and_inv not %[[AND]] : i1
  // CHECK: hw.output %[[AND]], %[[NOT]]
  %0 = synth.aig.and_inv %a, %b : i1
  // NOT(a) OR NOT(b) = NOT(NOT(NOT(a)) AND NOT(NOT(b))) = NOT(a AND b)
  %not_a = synth.aig.and_inv not %a : i1
  %not_b = synth.aig.and_inv not %b : i1
  // OR(x, y) = NOT(NOT(x) AND NOT(y))
  %and_nots = synth.aig.and_inv not %not_a, not %not_b : i1
  %1 = synth.aig.and_inv not %and_nots : i1
  hw.output %0, %1 : i1, i1
}

// Test with MajorityInverterOp
// CHECK-LABEL: hw.module @test_mig_equiv
hw.module @test_mig_equiv(in %a: i1, in %b: i1, in %c: i1, out out1: i1, out out2: i1) {
  // Two equivalent MAJ operations with permuted inputs
  // MAJ is symmetric, so MAJ(a,b,c) = MAJ(c,b,a)
  // CHECK: %[[MAJ:.+]] = synth.mig.maj_inv %a, %b, %c : i1
  // CHECK: hw.output %[[MAJ]], %[[MAJ]]
  %0 = synth.mig.maj_inv %a, %b, %c : i1
  %1 = synth.mig.maj_inv %c, %b, %a : i1
  hw.output %0, %1 : i1, i1
}

// Test that non-equivalent nodes are not merged
// CHECK-LABEL: hw.module @test_non_equiv
hw.module @test_non_equiv(in %a: i1, in %b: i1, out out1: i1, out out2: i1) {
  // These are different functions and should NOT be merged
  // CHECK: %[[AND1:.+]] = synth.aig.and_inv %a, %b : i1
  // CHECK: %[[AND2:.+]] = synth.aig.and_inv %a, not %b : i1
  // CHECK: hw.output %[[AND1]], %[[AND2]]
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %a, not %b : i1
  hw.output %0, %1 : i1, i1
}

// Test mixed AIG equivalence
// AND(a, AND(b, c)) is equivalent to AND(a, b, c)
// CHECK-LABEL: hw.module @test_mixed
hw.module @test_mixed(in %a: i1, in %b: i1, in %c: i1, out out1: i1, out out2: i1) {
  // Both compute a AND b AND c
  // CHECK: synth.aig.and_inv
  // CHECK: synth.aig.and_inv
  // Output should use the same value for both
  %bc = synth.aig.and_inv %b, %c : i1
  %0 = synth.aig.and_inv %a, %bc : i1
  %1 = synth.aig.and_inv %a, %b, %c : i1
  hw.output %0, %1 : i1, i1
}
