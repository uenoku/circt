// REQUIRES: z3-integration
// RUN: circt-opt %s --synth-cut-rewrite='db-files=%S/../../test/Dialect/Synth/Inputs/cut-rewrite-db.mlir strategy=area' -o %t
// RUN: circt-lec.sh %s %t --c1 canonical --c2 canonical
// RUN: circt-lec.sh %s %t --c1 input_phase --c2 input_phase
// RUN: circt-lec.sh %s %t --c1 output_phase --c2 output_phase

hw.module @canonical(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

hw.module @input_phase(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @output_phase(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}
