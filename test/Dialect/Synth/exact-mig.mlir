// RUN: not circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite))' 2>&1 | FileCheck %s --check-prefix=ERR

// ERR: synth-cut-rewrite requires 'db-file'
hw.module @missing_db_file(in %a : i1, out y : i1) {
  %0 = synth.aig.and_inv %a : i1
  hw.output %0 : i1
}
