// RUN: circt-opt %s --verify-diagnostics --split-input-file | FileCheck %s

hw.module @bundle_valid(in %a: i1, in %b: i1, out o0: i1, out o1: i1) {
  // CHECK-LABEL: hw.module @bundle_valid
  // CHECK: synth.create_bundle %a, %b : (i1, i1) -> (i1, i1)
  %o0, %o1 = synth.create_bundle %a, %b : (i1, i1) -> (i1, i1)
  hw.output %o0, %o1 : i1, i1
}

// -----

hw.module @bundle_arity(in %a: i1, in %b: i1, out o0: i1) {
  // expected-error @+1 {{'synth.create_bundle' op requires as many results as operands}}
  %o0 = synth.create_bundle %a, %b : (i1, i1) -> i1
  hw.output %o0 : i1
}

// -----

hw.module @bundle_types(in %a: i1, in %b: i1, out o0: i1, out o1: i8) {
  // expected-error @+1 {{'synth.create_bundle' op result 1 type must match operand type}}
  %o0, %o1 = synth.create_bundle %a, %b : (i1, i1) -> (i1, i8)
  hw.output %o0, %o1 : i1, i8
}
