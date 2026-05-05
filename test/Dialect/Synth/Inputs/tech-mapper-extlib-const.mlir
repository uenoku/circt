hw.module @ext_identity_with_const_one(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %const_one = hw.constant 1 : i1
  %0 = synth.aig.and_inv %a, %const_one : i1
  hw.output %0 : i1
}

hw.module @ext_identity_with_const_zero(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %const_zero = hw.constant 0 : i1
  %0 = synth.aig.and_inv %a, %const_zero : i1
  hw.output %0 : i1
}
