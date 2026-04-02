// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=mig-exact --max-inputs=2 --sat-solver=z3 | FileCheck %s --check-prefix=MIG
// RUN: circt-synth-dbgen --kind=aig-exact --max-inputs=2 --sat-solver=z3 | FileCheck %s --check-prefix=AIG

// MIG: hw.module @mig_exact_i1_tt_0
// MIG-SAME: attributes {hw.techlib.info =
// MIG-SAME: synth.cut_rewrite.inverter_kind = "mig"
// MIG: synth.cut_rewrite.inverter_kind = "mig", synth.cut_rewrite.canonical_tt = 0 : i2

// AIG: hw.module @aig_exact_i1_tt_0
// AIG-SAME: attributes {hw.techlib.info =
// AIG-SAME: synth.cut_rewrite.inverter_kind = "aig"
// AIG: synth.cut_rewrite.inverter_kind = "mig", synth.cut_rewrite.canonical_tt = 0 : i2

// Empty file; the generator does not consume input IR.
