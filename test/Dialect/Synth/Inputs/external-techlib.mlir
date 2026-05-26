hw.module @ext_and(in %a : i1, in %b : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<"Y", "a", 1, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "b", 1, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}
