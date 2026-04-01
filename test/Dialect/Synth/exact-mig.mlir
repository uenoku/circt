// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db=bogus}))' -verify-diagnostics

// expected-error @+1 {{unknown cut rewrite database 'bogus'}}
hw.module @invalid_db(in %a : i1, out y : i1) {
  %0 = synth.aig.and_inv %a : i1
  hw.output %0 : i1
}
