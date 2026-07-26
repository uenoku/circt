// RUN: not circt-opt %s --synth-cut-rewrite 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not circt-opt %s --synth-cut-rewrite='db-files=%S/Inputs/cut-rewrite-invalid-db.mlir' 2>&1 | FileCheck %s --check-prefix=BADOP

// MISSING: synth-cut-rewrite requires at least one 'db-files' entry
// BADOP: cut-rewrite database bodies only support hw.constant and BooleanLogicOpInterface operations

hw.module @test(in %a : i1, out y : i1) {
  hw.output %a : i1
}
