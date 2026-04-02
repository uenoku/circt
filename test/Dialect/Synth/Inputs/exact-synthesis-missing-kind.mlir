module {
  hw.module @bogus(in %i0 : i1, out y : i1)
      attributes {hw.techlib.info = {area = 0.000000e+00 : f64, delay = [[0]]},
                  synth.cut_rewrite.canonical_tt = 0 : i2} {
    %false = hw.constant false
    hw.output %false : i1
  }
}
