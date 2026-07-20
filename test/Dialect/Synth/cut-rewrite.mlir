// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db-files=%S/Inputs/cut-rewrite-dot-db.mlir test=true}))' %s | FileCheck %s --check-prefix=DOT
// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(synth-cut-rewrite{db-files=%S/Inputs/cut-rewrite-xag-db.mlir test=true}))' %s | FileCheck %s --check-prefix=XAG

// DOT-LABEL: hw.module @rewrite_xor_to_dot
// DOT-NEXT: %[[DOT:.+]] = synth.dot not %b, %b, not %a {test.arrival_times = [1]} : i1
// DOT-NEXT: hw.output %[[DOT]] : i1
hw.module @rewrite_xor_to_dot(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.xor_inv %a, %b : i1
  hw.output %0 : i1
}

// XAG-LABEL: hw.module @rewrite_dot_to_xag
// XAG-NEXT: %[[AB:.+]] = synth.aig.and_inv %a, %b : i1
// XAG-NEXT: %[[ORLIKE:.+]] = synth.aig.and_inv not %c, not %[[AB]] : i1
// XAG-NEXT: %[[XOR:.+]] = synth.xor_inv %a, %[[ORLIKE]] : i1
// XAG-NEXT: %[[NOT:.+]] = synth.aig.and_inv not %[[XOR]] {test.arrival_times = [3]} : i1
// XAG-NEXT: hw.output %[[NOT]] : i1
hw.module @rewrite_dot_to_xag(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = synth.dot %a, %b, %c : i1
  hw.output %0 : i1
}
