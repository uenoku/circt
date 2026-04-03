// RUN: circt-synth-dbgen --kind=npn --max-inputs=2 | FileCheck %s --check-prefix=NPN

// NPN: hw.module @npn_i1_tt_0_v0
// NPN: %[[TT:.+]] = comb.truth_table %i0 -> [false, false]
// NPN: hw.output %[[TT]] : i1

// Empty file; the generator does not consume input IR.
