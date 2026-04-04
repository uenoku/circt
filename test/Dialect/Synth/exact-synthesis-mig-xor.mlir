// REQUIRES: z3-integration
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig-xor sat-solver=z3})' | FileCheck %s --check-prefix=AREA
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig-xor objective=depth-size sat-solver=z3})' | FileCheck %s --check-prefix=DEPTH

// AREA-LABEL: hw.module @xor_tt
// AREA: %[[XOR:.+]] = comb.xor %a, %b : i1
// AREA-NEXT: hw.output %[[XOR]] : i1

// DEPTH-LABEL: hw.module @xor_tt
// DEPTH: %[[XOR:.+]] = comb.xor %a, %b : i1
// DEPTH-NEXT: hw.output %[[XOR]] : i1
hw.module @xor_tt(in %a : i1, in %b : i1, out y : i1) {
  %y = comb.truth_table %b, %a -> [false, true, true, false]
  hw.output %y : i1
}

// AREA-LABEL: hw.module @and_tt
// AREA: %[[FALSE:.+]] = hw.constant false
// AREA-NEXT: %[[AND:.+]] = synth.mig.maj_inv %[[FALSE]], %i0, %i1 : i1
// AREA-NEXT: hw.output %[[AND]] : i1

// DEPTH-LABEL: hw.module @and_tt
// DEPTH: %[[FALSE:.+]] = hw.constant false
// DEPTH-NEXT: %[[AND:.+]] = synth.mig.maj_inv %[[FALSE]], %i0, %i1 : i1
// DEPTH-NEXT: hw.output %[[AND]] : i1
hw.module @and_tt(in %i0 : i1, in %i1 : i1, out y : i1) {
  %y = comb.truth_table %i1, %i0 -> [false, false, false, true]
  hw.output %y : i1
}

// AREA-LABEL: hw.module @maj_xor_tt
// AREA-DAG: comb.xor
// AREA-DAG: synth.mig.maj_inv
// AREA: hw.output

// DEPTH-LABEL: hw.module @maj_xor_tt
// DEPTH-DAG: comb.xor
// DEPTH-DAG: synth.mig.maj_inv
// DEPTH: hw.output
hw.module @maj_xor_tt(in %a : i1, in %b : i1, in %c : i1, in %d : i1,
                      out y : i1) {
  %y = comb.truth_table %d, %c, %b, %a -> [false, false, false, true,
                                           false, true, true, true,
                                           true, true, true, false,
                                           true, false, false, false]
  hw.output %y : i1
}
