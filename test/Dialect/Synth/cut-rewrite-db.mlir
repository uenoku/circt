// RUN: circt-opt %s --synth-cut-rewrite='db-files=%S/Inputs/cut-rewrite-db.mlir strategy=area test' | FileCheck %s

hw.module @canonical(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @canonical
// CHECK: %[[DOT:.+]] = synth.dot not %b, not %a, %b : i1
// CHECK: %[[INV:.+]] = synth.aig.and_inv not %[[DOT]] {test.arrival_times = [1]} : i1
// CHECK: hw.output %[[INV]] : i1

// Exercise input NPN phases, not just the canonical representative.
hw.module @phase_adjusted(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

// CHECK-LABEL: hw.module @phase_adjusted
// CHECK: synth.aig.and_inv not %
// CHECK: synth.dot
// CHECK: hw.output

// Exercise output NPN phase materialization.
hw.module @output_phase(in %a : i1, in %b : i1, out y : i1) {
  %0 = synth.aig.and_inv not %a, not %b : i1
  %1 = synth.aig.and_inv not %0 : i1
  hw.output %1 : i1
}

// CHECK-LABEL: hw.module @output_phase
// CHECK: %[[ODOT:.+]] = synth.dot
// CHECK: %[[ONOR:.+]] = synth.aig.and_inv not %[[ODOT]]
// CHECK: %[[OOR:.+]] = synth.aig.and_inv not %[[ONOR]] {test.arrival_times = [1]} : i1
// CHECK: hw.output %[[OOR]] : i1
