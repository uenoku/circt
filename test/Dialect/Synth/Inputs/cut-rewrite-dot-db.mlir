module {
  hw.module @dot_cell(in %i0 : i1, in %i1 : i1, out y : i1) {
    %0 = synth.dot not %i1, %i1, not %i0 : i1
    hw.output %0 : i1
  }

  synth.cut_rewrite_pattern (%i0: i1, %i1: i1) -> i1 attributes {
    allow_negation = true,
    cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [
      #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>,
      #synth.linear_timing_arc<1, 0, #synth.polarity<positive>>
    ]>
  } {
    %0 = hw.instance "mapped" @dot_cell(i0: %i0: i1, i1: %i1: i1) -> (y: i1)
    synth.yield %0 : i1
  }
}
