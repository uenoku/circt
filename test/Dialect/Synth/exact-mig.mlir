// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-exact-mig{sat-solver=bogus}))' -verify-diagnostics

// expected-error @+1 {{unknown SAT solver backend 'bogus'}}
hw.module @invalid_backend(in %a : i1, out y : i1) {
  %0 = synth.aig.and_inv %a : i1
  hw.output %0 : i1
}
