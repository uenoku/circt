// REQUIRES: z3-integration, libz3
// REQUIRES: circt-lec-jit
// RUN: circt-synth-dbgen --kind=npn --max-inputs=3 -o %t.pre.mlir
// RUN: circt-opt %t.pre.mlir -pass-pipeline='builtin.module(synth-exact-synthesis{kind=mig-xor sat-solver=z3})' -o %t.db.mlir
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-greedy-cut-rewrite{db-files=%t.pre.mlir,%t.db.mlir max-iterations=1}))' -o %t.mlir
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 majority_tree --c2 majority_tree | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 no_change_and --c2 no_change_and | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 xor_from_aig --c2 xor_from_aig | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 already_optimal --c2 already_optimal | FileCheck %s --check-prefix=LEC
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 xor_and_mix --c2 xor_and_mix | FileCheck %s --check-prefix=LEC

// LEC: c1 == c2
hw.module @majority_tree(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %ab = synth.aig.and_inv %a, %b : i1
  %ac = synth.aig.and_inv %a, %c : i1
  %or0 = synth.aig.and_inv not %ab, not %ac : i1
  %bc = synth.aig.and_inv %b, %c : i1
  %y = synth.aig.and_inv not %or0, not %bc : i1
  hw.output %y : i1
}

hw.module @no_change_and(in %a : i1, in %b : i1, out y : i1) {
  %y = synth.aig.and_inv %a, %b : i1
  hw.output %y : i1
}

hw.module @xor_from_aig(in %a : i1, in %b : i1, out y : i1) {
  %lhs = synth.aig.and_inv %a, not %b : i1
  %rhs = synth.aig.and_inv not %a, %b : i1
  %y = synth.aig.and_inv not %lhs, not %rhs : i1
  hw.output %y : i1
}

hw.module @already_optimal(in %a : i1, in %b : i1, out y : i1) {
  %false = hw.constant false
  %0 = synth.mig.maj_inv %false, %a, not %b : i1
  %1 = synth.mig.maj_inv %false, not %a, not %b : i1
  %2 = synth.mig.maj_inv not %a, %0, not %1 : i1
  hw.output %2 : i1
}

hw.module @xor_and_mix(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %lhs = synth.aig.and_inv %a, not %b : i1
  %rhs = synth.aig.and_inv not %a, %b : i1
  %xor = synth.aig.and_inv not %lhs, not %rhs : i1
  %y = comb.xor %xor, %c : i1
  hw.output %y : i1
}
