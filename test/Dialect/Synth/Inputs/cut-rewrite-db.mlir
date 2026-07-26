module {
  // A unary-capable Boolean operation is used to materialize NPN phases.
  hw.module @npn_i1_tt_1(in %i0 : i1, out y : i1) {
    %0 = synth.aig.and_inv not %i0 : i1
    hw.output %0 : i1
  }

  // Canonical representative 0x1 of the two-input AND NPN class.
  hw.module @npn_i2_tt_1(in %i0 : i1, in %i1 : i1, out y : i1) {
    %0 = synth.dot not %i1, not %i0, %i1 : i1
    %1 = synth.aig.and_inv not %0 : i1
    hw.output %1 : i1
  }
}
