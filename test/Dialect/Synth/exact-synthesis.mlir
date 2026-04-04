// REQUIRES: z3-integration
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig sat-solver=z3})' | FileCheck %s --check-prefix=MIG
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig objective=depth-size sat-solver=z3})' | FileCheck %s --check-prefix=MIG-DEPTH

// MIG-LABEL: hw.module @and_tt
// MIG: %[[FALSE:.+]] = hw.constant false
// MIG-NEXT: %[[AND:.+]] = synth.mig.maj_inv %[[FALSE]], %i0, %i1 : i1
// MIG-NEXT: hw.output %[[AND]] : i1

// MIG-DEPTH-LABEL: hw.module @and_tt
// MIG-DEPTH: %[[FALSE:.+]] = hw.constant false
// MIG-DEPTH-NEXT: %[[AND:.+]] = synth.mig.maj_inv %[[FALSE]], %i0, %i1 : i1
// MIG-DEPTH-NEXT: hw.output %[[AND]] : i1
hw.module @and_tt(in %i0 : i1, in %i1 : i1, out y : i1) {
  %y = comb.truth_table %i1, %i0 -> [false, false, false, true]
  hw.output %y : i1
}
