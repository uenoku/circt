// RUN: not circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db=bogus}))' 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: unknown cut rewrite database 'bogus'
hw.module @invalid_db(in %a : i1, out y : i1) {
  %0 = synth.aig.and_inv %a : i1
  hw.output %0 : i1
}
