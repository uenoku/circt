// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=mig-exact --max-inputs=2 --sat-solver=z3 | FileCheck %s

// CHECK: hw.module @mig_exact_i1_tt_0
// CHECK-SAME: attributes {hw.techlib.info =
// CHECK: synth.cut_rewrite.canonical_tt = 0 : i2
// CHECK: synth.cut_rewrite.db = "MIG_EXACT"

// Empty file; the generator does not consume input IR.
