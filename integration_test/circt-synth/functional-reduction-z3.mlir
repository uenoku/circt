// REQUIRES: z3-integration
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-functional-reduction{num-random-patterns=64}))' | FileCheck %s

// SAT should prove that AND(AND(a, not b), AND(c, not d)) is equivalent to
// AND(a, not b, c, not d), and the pass should materialize that with a choice.
// CHECK-LABEL: hw.module @functional_reduction_sat
hw.module @functional_reduction_sat(in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                                    out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: %[[AB:.+]] = synth.aig.and_inv %a, not %b : i1
  // CHECK-NEXT: %[[CD:.+]] = synth.aig.and_inv %c, not %d : i1
  // CHECK-NEXT: %[[TREE:.+]] = synth.aig.and_inv %[[AB]], %[[CD]] : i1
  // CHECK-NEXT: %[[FLAT:.+]] = synth.aig.and_inv %a, not %b, %c, not %d : i1
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[TREE]], %[[FLAT]]
  // CHECK-NEXT: %[[DIFF:.+]] = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  // CHECK-NEXT: hw.output %[[CHOICE]], %[[CHOICE]], %[[DIFF]] : i1, i1, i1
  %0 = synth.aig.and_inv %a, not %b : i1
  %1 = synth.aig.and_inv %c, not %d : i1
  %2 = synth.aig.and_inv %0, %1 : i1
  %3 = synth.aig.and_inv %a, not %b, %c, not %d : i1
  %4 = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  hw.output %2, %3, %4 : i1, i1, i1
}

// SAT should also prove equivalence across the newly supported comb and MIG
// nodes.
// CHECK-LABEL: hw.module @functional_reduction_supported_ops_sat
hw.module @functional_reduction_supported_ops_sat(
    in %a: i1, in %b: i1, in %c: i1,
    out out0: i1, out out1: i1, out out2: i1, out out3: i1,
    out out4: i1, out out5: i1, out out6: i1, out out7: i1) {
  // CHECK: %[[FALSE:.+]] = hw.constant false
  // CHECK: %[[OR0:.+]] = comb.or %a, %b, %c : i1
  // CHECK-NEXT: %[[OR1:.+]] = comb.or %c, %b, %a : i1
  // CHECK-NEXT: %[[ORCHOICE:.+]] = synth.choice %[[OR0]], %[[OR1]] : i1
  // CHECK: %[[AND0:.+]] = comb.and %a, %b : i1
  // CHECK-NEXT: %[[AND1:.+]] = synth.mig.maj_inv %a, %b, %[[FALSE]] : i1
  // CHECK-NEXT: %[[ANDCHOICE:.+]] = synth.choice %[[AND0]], %[[AND1]] : i1
  // CHECK: %[[XOR0:.+]] = comb.xor %a, %b : i1
  // CHECK-NEXT: %[[XOR1:.+]] = comb.xor %b, %a : i1
  // CHECK-NEXT: %[[XORCHOICE:.+]] = synth.choice %[[XOR0]], %[[XOR1]] : i1
  // CHECK: %[[MAJ0:.+]] = synth.mig.maj_inv %a, %b, %c : i1
  // CHECK: %[[CXOR:.+]] = comb.and %c, %[[XORCHOICE]] : i1
  // CHECK: %[[MAJ1:.+]] = comb.or %[[ANDCHOICE]], %[[CXOR]] : i1
  // CHECK-NEXT: %[[MAJCHOICE:.+]] = synth.choice %[[MAJ0]], %[[MAJ1]] : i1
  // CHECK: hw.output %[[ORCHOICE]], %[[ORCHOICE]], %[[ANDCHOICE]], %[[ANDCHOICE]], %[[XORCHOICE]], %[[XORCHOICE]], %[[MAJCHOICE]], %[[MAJCHOICE]] : i1, i1, i1, i1, i1, i1, i1, i1
  %false = hw.constant false
  %0 = comb.or %a, %b, %c : i1
  %1 = comb.or %c, %b, %a : i1
  %2 = comb.and %a, %b : i1
  %3 = synth.mig.maj_inv %a, %b, %false : i1
  %4 = comb.xor %a, %b : i1
  %5 = comb.xor %b, %a : i1
  %6 = synth.mig.maj_inv %a, %b, %c : i1
  %7 = comb.and %c, %4 : i1
  %8 = comb.or %2, %7 : i1
  hw.output %0, %1, %2, %3, %4, %5, %6, %8 : i1, i1, i1, i1, i1, i1, i1, i1
}

// SAT should prove equivalence for variadic 5-input majority gates as well.
// CHECK-LABEL: hw.module @functional_reduction_five_input_mig_sat
hw.module @functional_reduction_five_input_mig_sat(
    in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1,
    out out0: i1, out out1: i1) {
  // CHECK: %[[M0:.+]] = synth.mig.maj_inv %a, %b, %c, %d, %e : i1
  // CHECK-NEXT: %[[M1:.+]] = synth.mig.maj_inv %e, %d, %c, %b, %a : i1
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[M0]], %[[M1]] : i1
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]] : i1, i1
  %0 = synth.mig.maj_inv %a, %b, %c, %d, %e : i1
  %1 = synth.mig.maj_inv %e, %d, %c, %b, %a : i1
  hw.output %0, %1 : i1, i1
}
