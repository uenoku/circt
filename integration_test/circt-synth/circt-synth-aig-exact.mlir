// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=aig-exact --max-inputs=2 --sat-solver=z3 -o %t.db.mlir
// RUN: circt-synth %s --target-ir=mig --cut-rewrite-db-file=%t.db.mlir --until-before mapping | FileCheck %s

// CHECK-LABEL: hw.module @and_from_mig
// CHECK-NOT: synth.mig.maj_inv
// CHECK: %[[AND:.+]] = synth.aig.and_inv %a, %b : i1
// CHECK: hw.output
hw.module @and_from_mig(in %a : i1, in %b : i1, out y : i1) {
  %y = comb.and %a, %b : i1
  hw.output %y : i1
}
