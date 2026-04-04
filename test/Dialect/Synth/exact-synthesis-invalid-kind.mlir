// REQUIRES: z3-integration
// RUN: not circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{kind=aig sat-solver=z3})' --verify-diagnostics

// expected-error @+1 {{unsupported exact synthesis kind 'aig'}}
hw.module @and_tt(in %i0 : i1, in %i1 : i1, out y : i1) {
  %y = comb.truth_table %i1, %i0 -> [false, false, false, true]
  hw.output %y : i1
}
