module {
  synth.cut_rewrite_pattern (%i0: i1, %i1: i1, %i2: i1) -> i1 attributes {
    cost = #synth.mapping_cost<area = 3.0 : f64, arcs = [
      #synth.linear_timing_arc<3, 0, #synth.polarity<positive>>,
      #synth.linear_timing_arc<2, 0, #synth.polarity<positive>>,
      #synth.linear_timing_arc<3, 0, #synth.polarity<positive>>
    ]>
  } {
    %0 = synth.aig.and_inv %i0, %i1 : i1
    %1 = synth.aig.and_inv not %i2, not %0 : i1
    %2 = synth.xor_inv %i0, %1 : i1
    synth.yield %2 : i1
  }
}
