hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_3(in %a : i1, in %b : i1, in %c : i1, out result : i1) attributes {hw.techlib.info = {area = 10.0 : f64, delay = [[1], [1], [1]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}
