module {
  synth.cut_rewrite_pattern (%i0: i1, %i1: i1) -> i1 attributes {
    cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
      #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
      #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
    ]>
  } {
    %0 = synth.dot not %i1, %i1, not %i0 : i1
    synth.yield %0 : i1
  }
}
