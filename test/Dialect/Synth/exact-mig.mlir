// RUN: not circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite))' 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not circt-opt %s --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db-file=%S/Inputs/exact-synthesis-missing-kind.mlir}))' 2>&1 | FileCheck %s --check-prefix=DBERR

// ERR: synth-cut-rewrite requires 'db-file'
// DBERR: cut-rewrite database module missing 'synth.cut_rewrite.inverter_kind'
hw.module @missing_db_file(in %a : i1, out y : i1) {
  %0 = synth.aig.and_inv %a : i1
  hw.output %0 : i1
}
