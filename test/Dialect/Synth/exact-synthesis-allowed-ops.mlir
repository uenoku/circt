// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:3})' | FileCheck %s --check-prefix=AND3
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.xor_inv:3})' | FileCheck %s --check-prefix=XOR3
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot:3})' | FileCheck %s --check-prefix=DOT
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis)' 2>&1 | FileCheck %s --check-prefix=NO-OPS
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.dot})' 2>&1 | FileCheck %s --check-prefix=MISSING-ARITY

// NO-OPS: synth-exact-synthesis requires at least one 'allowed-ops=name:arity' entry
// MISSING-ARITY: expected explicit arity for 'synth.dot', e.g. 'synth.dot:3'

// AND3-LABEL: hw.module @and3_tt
// AND3: %[[AND:.+]] = synth.aig.and_inv %c, %b, %a : i1
// AND3-NEXT: hw.output %[[AND]] : i1
hw.module @and3_tt(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = comb.truth_table %a, %b, %c -> [false, false, false, false,
                                        false, false, false, true]
  hw.output %0 : i1
}

// XOR3-LABEL: hw.module @xor3_tt
// XOR3: %[[XOR:.+]] = synth.xor_inv{{.*}}: i1
// XOR3-NEXT: hw.output %[[XOR]] : i1
hw.module @xor3_tt(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = comb.truth_table %a, %b, %c -> [false, true, true, false,
                                        true, false, false, true]
  hw.output %0 : i1
}

// DOT-LABEL: hw.module @dot_tt
// DOT: %[[DOT:.+]] = synth.dot{{.*}}: i1
// DOT-NEXT: hw.output %[[DOT]] : i1
hw.module @dot_tt(in %x : i1, in %y : i1, in %z : i1, out y : i1) {
  %0 = comb.truth_table %x, %y, %z -> [false, true, false, true,
                                        true, false, false, false]
  hw.output %0 : i1
}
