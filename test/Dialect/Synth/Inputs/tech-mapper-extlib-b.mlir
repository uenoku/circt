hw.module @ext_two_not(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %0 = synth.aig.and_inv %a, not %b : i1
  hw.output %0 : i1
}
