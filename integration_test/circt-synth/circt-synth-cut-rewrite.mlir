// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=mig-exact --max-inputs=3 --sat-solver=z3 -o %t.db.mlir
// RUN: circt-synth %s --cut-rewrite-db=mig-exact --cut-rewrite-db-file=%t.db.mlir --until-before mapping | FileCheck %s

// CHECK-LABEL: hw.module @majority_tree
// CHECK: %[[FALSE:.+]] = hw.constant false
// CHECK-NEXT: %[[M0:.+]] = synth.mig.maj_inv
// CHECK-NEXT: %[[M1:.+]] = synth.mig.maj_inv
// CHECK-NEXT: %[[M2:.+]] = synth.mig.maj_inv
// CHECK-NOT: synth.aig.and_inv %a, %b
// CHECK-NEXT: hw.output %[[M2]] : i1
hw.module @majority_tree(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %ab = synth.aig.and_inv %a, %b : i1
  %ac = synth.aig.and_inv %a, %c : i1
  %or0 = synth.aig.and_inv not %ab, not %ac : i1
  %bc = synth.aig.and_inv %b, %c : i1
  %y = synth.aig.and_inv not %or0, not %bc : i1
  hw.output %y : i1
}
