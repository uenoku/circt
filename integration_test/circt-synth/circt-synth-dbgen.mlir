// REQUIRES: z3-integration
// RUN: circt-synth-dbgen --kind=mig-exact --max-inputs=2 --sat-solver=z3 | FileCheck %s --check-prefix=MIG
// RUN: circt-synth-dbgen --kind=aig-exact --max-inputs=2 --sat-solver=z3 | FileCheck %s --check-prefix=AIG

// MIG: hw.module @mig_exact_i1_tt_0
// MIG-SAME: attributes {synth.cut_rewrite.inverter_kind = "mig"}

// AIG: hw.module @aig_exact_i1_tt_0
// AIG-SAME: attributes {synth.cut_rewrite.inverter_kind = "aig"}

// Empty file; the generator does not consume input IR.
