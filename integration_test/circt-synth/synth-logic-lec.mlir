// REQUIRES: libz3
// REQUIRES: circt-lec-jit

// RUN: circt-opt %s --convert-synth-to-comb -o %t.lowered.mlir
// RUN: circt-lec %S/Inputs/synth-logic-lec-ref.mlir %t.lowered.mlir -c1=xor_inv_ref -c2=xor_inv_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=XOR
// RUN: circt-lec %S/Inputs/synth-logic-lec-ref.mlir %t.lowered.mlir -c1=dot_ref -c2=dot_impl --shared-libs=%libz3 | FileCheck %s --check-prefix=DOT
// RUN: circt-opt %S/Inputs/synth-logic-lec-ref.mlir --convert-comb-to-synth -o %t.aig.mlir
// RUN: circt-lec %S/Inputs/synth-logic-lec-ref.mlir %t.aig.mlir -c1=xor_inv_ref -c2=xor_inv_ref --shared-libs=%libz3 | FileCheck %s --check-prefix=AIG

// XOR: c1 == c2
// DOT: c1 == c2
// AIG: c1 == c2

hw.module @xor_inv_impl(in %a : i4, in %b : i4, in %c : i4, out out : i4) {
  %0 = synth.xor_inv %a, not %b, %c : i4
  hw.output %0 : i4
}

hw.module @dot_impl(in %x : i4, in %y : i4, in %z : i4, out out : i4) {
  %0 = synth.dot not %x, %y, %z : i4
  hw.output %0 : i4
}
