R"mlir(
// Generated from mockturtle experiments/cell_libraries/asap7.genlib.
// Constants are omitted; synthesis keeps constants in IR.
hw.module @BUFx2_ASAP7_75t_R(in %A : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.07 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 1978, 0, #synth.polarity<positive>>], input_caps = {}>} {
  hw.output %A : i1
}

hw.module @INVx1_ASAP7_75t_R(in %A : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.04 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 1477, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A : i1
  hw.output %n0 : i1
}

hw.module @AND2x2_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2269, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2273, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  hw.output %n0 : i1
}

hw.module @OR2x4_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2510, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2504, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @NAND2x1_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2302, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2215, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @NOR2x1_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2393, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2319, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B : i1
  hw.output %n0 : i1
}

hw.module @XNOR2x2_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 3692, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3652, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  %n1 = synth.aig.and_inv not %A, not %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @XOR2x2_ASAP7_75t_R(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 3591, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3634, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, not %B : i1
  %n1 = synth.aig.and_inv not %A, %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AND3x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2564, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2602, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2648, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C : i1
  hw.output %n0 : i1
}

hw.module @OR3x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2806, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2854, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2896, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @AO21x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.09 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2172, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2209, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2112, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @OA21x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.1 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2162, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2198, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 1974, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B : i1
  %n1 = synth.aig.and_inv %A2, %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @MAJx2_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.13 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2770, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3312, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 3034, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  %n1 = synth.aig.and_inv %A, %C : i1
  %n2 = synth.aig.and_inv %B, %C : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @AOI21x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2204, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2015, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2167, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B : i1
  %n1 = synth.aig.and_inv not %A2, not %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @OAI21x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2173, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 1993, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 1870, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %n0, %B : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @NAND3x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2757, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2889, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2889, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @NOR3x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2914, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3056, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 3061, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C : i1
  hw.output %n0 : i1
}

hw.module @AO22x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.13 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2274, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2352, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2275, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2343, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @OA211x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2527, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2573, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2670, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2718, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B, %C : i1
  %n1 = synth.aig.and_inv %A2, %B, %C : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @OR4x2_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2901, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2994, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 3077, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 3134, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, not %D : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @OA22x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.15 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2239, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2279, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2222, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2272, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A1, %B2 : i1
  %n2 = synth.aig.and_inv %A2, %B1 : i1
  %n3 = synth.aig.and_inv %A2, %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @AND4x2_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 2732, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2769, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2826, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 2869, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C, %D : i1
  hw.output %n0 : i1
}

hw.module @OA31x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.22 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2620, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2563, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 2642, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2488, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A2, %B1 : i1
  %n2 = synth.aig.and_inv %A3, %B1 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @AO211x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.23 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2603, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2503, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2432, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2378, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B, not %C : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @AOI211x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2778, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3008, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2358, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2433, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B, not %C : i1
  %n1 = synth.aig.and_inv not %A2, not %B, not %C : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AO31x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.23 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2313, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2295, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 2302, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 2420, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv not %n0, not %B : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @OAI22x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.15 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2224, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2401, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2193, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2364, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %B1, not %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AOI22x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.15 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2229, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2404, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2193, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2375, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A1, not %B2 : i1
  %n2 = synth.aig.and_inv not %A2, not %B1 : i1
  %n3 = synth.aig.and_inv not %A2, not %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @AO32x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2650, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2592, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 2683, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2720, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2624, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AND5x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, in %D : i1, in %E : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 3193, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3350, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 3520, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 3622, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "E", 3685, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C, %D, %E : i1
  hw.output %n0 : i1
}

hw.module @OR5x1_ASAP7_75t_R(in %A : i1, in %B : i1, in %C : i1, in %D : i1, in %E : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.12 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 3179, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 3327, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 3482, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 3585, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "E", 3664, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, not %D, not %E : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @AO221x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2818, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2748, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2704, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2620, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2549, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1, not %C : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AOI221x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.2 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3075, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2928, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2921, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2800, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2654, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1, not %C : i1
  %n1 = synth.aig.and_inv not %A1, not %B2, not %C : i1
  %n2 = synth.aig.and_inv not %A2, not %B1, not %C : i1
  %n3 = synth.aig.and_inv not %A2, not %B2, not %C : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @OA221x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.23 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2770, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2851, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2580, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2671, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2480, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C : i1
  %n1 = synth.aig.and_inv %A1, %B2, %C : i1
  %n2 = synth.aig.and_inv %A2, %B1, %C : i1
  %n3 = synth.aig.and_inv %A2, %B2, %C : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @AO33x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.15 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2944, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2925, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 2995, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2921, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2906, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 2950, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2, %B3 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @OA33x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.15 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2859, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2813, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 2811, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2730, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2703, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 2706, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A1, %B2 : i1
  %n2 = synth.aig.and_inv %A1, %B3 : i1
  %n3 = synth.aig.and_inv %A2, %B1 : i1
  %n4 = synth.aig.and_inv %A2, %B2 : i1
  %n5 = synth.aig.and_inv %A2, %B3 : i1
  %n6 = synth.aig.and_inv %A3, %B1 : i1
  %n7 = synth.aig.and_inv %A3, %B2 : i1
  %n8 = synth.aig.and_inv %A3, %B3 : i1
  %n9 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5, not %n6, not %n7, not %n8 : i1
  %n10 = synth.aig.and_inv not %n9 : i1
  hw.output %n10 : i1
}

hw.module @AO222x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, in %C2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2978, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2944, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2919, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2865, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 2855, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 2791, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv %C1, %C2 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @OA222x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, in %C2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 2768, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 2725, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2804, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2770, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 2854, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 2848, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A1, %B1, %C2 : i1
  %n2 = synth.aig.and_inv %A1, %B2, %C1 : i1
  %n3 = synth.aig.and_inv %A1, %B2, %C2 : i1
  %n4 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n5 = synth.aig.and_inv %A2, %B1, %C2 : i1
  %n6 = synth.aig.and_inv %A2, %B2, %C1 : i1
  %n7 = synth.aig.and_inv %A2, %B2, %C2 : i1
  %n8 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5, not %n6, not %n7 : i1
  %n9 = synth.aig.and_inv not %n8 : i1
  hw.output %n9 : i1
}

hw.module @AO322x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, in %C2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.22 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3287, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3274, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3304, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3291, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 3232, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 3181, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 3108, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv %C1, %C2 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @OA331x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3121, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3121, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3181, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 2956, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2945, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 2989, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 2600, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A1, %B2, %C1 : i1
  %n2 = synth.aig.and_inv %A1, %B3, %C1 : i1
  %n3 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n4 = synth.aig.and_inv %A2, %B2, %C1 : i1
  %n5 = synth.aig.and_inv %A2, %B3, %C1 : i1
  %n6 = synth.aig.and_inv %A3, %B1, %C1 : i1
  %n7 = synth.aig.and_inv %A3, %B2, %C1 : i1
  %n8 = synth.aig.and_inv %A3, %B3, %C1 : i1
  %n9 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5, not %n6, not %n7, not %n8 : i1
  %n10 = synth.aig.and_inv not %n9 : i1
  hw.output %n10 : i1
}

hw.module @AO331x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.16 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3160, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3134, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3188, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3025, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 2990, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 3037, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 2769, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2, %B3 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1, not %C : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @AO332x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C1 : i1, in %C2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3387, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3360, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3420, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3247, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 3213, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 3258, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 3049, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 2971, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2, %B3 : i1
  %n2 = synth.aig.and_inv %C1, %C2 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @OA332x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C1 : i1, in %C2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3350, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3352, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3410, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3175, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 3165, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 3212, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 2921, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 2855, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A1, %B1, %C2 : i1
  %n2 = synth.aig.and_inv %A1, %B2, %C1 : i1
  %n3 = synth.aig.and_inv %A1, %B2, %C2 : i1
  %n4 = synth.aig.and_inv %A1, %B3, %C1 : i1
  %n5 = synth.aig.and_inv %A1, %B3, %C2 : i1
  %n6 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n7 = synth.aig.and_inv %A2, %B1, %C2 : i1
  %n8 = synth.aig.and_inv %A2, %B2, %C1 : i1
  %n9 = synth.aig.and_inv %A2, %B2, %C2 : i1
  %n10 = synth.aig.and_inv %A2, %B3, %C1 : i1
  %n11 = synth.aig.and_inv %A2, %B3, %C2 : i1
  %n12 = synth.aig.and_inv %A3, %B1, %C1 : i1
  %n13 = synth.aig.and_inv %A3, %B1, %C2 : i1
  %n14 = synth.aig.and_inv %A3, %B2, %C1 : i1
  %n15 = synth.aig.and_inv %A3, %B2, %C2 : i1
  %n16 = synth.aig.and_inv %A3, %B3, %C1 : i1
  %n17 = synth.aig.and_inv %A3, %B3, %C2 : i1
  %n18 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5, not %n6, not %n7, not %n8, not %n9, not %n10, not %n11, not %n12, not %n13, not %n14, not %n15, not %n16, not %n17 : i1
  %n19 = synth.aig.and_inv not %n18 : i1
  hw.output %n19 : i1
}

hw.module @AO333x1_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C1 : i1, in %C2 : i1, in %C3 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.17 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3679, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3647, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3711, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3546, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 3494, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 3534, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 3377, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 3330, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C3", 3375, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2, %B3 : i1
  %n2 = synth.aig.and_inv %C1, %C2, %C3 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @OA333x2_ASAP7_75t_R(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, in %B3 : i1, in %C1 : i1, in %C2 : i1, in %C3 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.19 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 3283, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 3259, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 3302, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 3433, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 3425, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B3", 3472, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 3613, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C2", 3614, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C3", 3669, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A1, %B1, %C2 : i1
  %n2 = synth.aig.and_inv %A1, %B1, %C3 : i1
  %n3 = synth.aig.and_inv %A1, %B2, %C1 : i1
  %n4 = synth.aig.and_inv %A1, %B2, %C2 : i1
  %n5 = synth.aig.and_inv %A1, %B2, %C3 : i1
  %n6 = synth.aig.and_inv %A1, %B3, %C1 : i1
  %n7 = synth.aig.and_inv %A1, %B3, %C2 : i1
  %n8 = synth.aig.and_inv %A1, %B3, %C3 : i1
  %n9 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n10 = synth.aig.and_inv %A2, %B1, %C2 : i1
  %n11 = synth.aig.and_inv %A2, %B1, %C3 : i1
  %n12 = synth.aig.and_inv %A2, %B2, %C1 : i1
  %n13 = synth.aig.and_inv %A2, %B2, %C2 : i1
  %n14 = synth.aig.and_inv %A2, %B2, %C3 : i1
  %n15 = synth.aig.and_inv %A2, %B3, %C1 : i1
  %n16 = synth.aig.and_inv %A2, %B3, %C2 : i1
  %n17 = synth.aig.and_inv %A2, %B3, %C3 : i1
  %n18 = synth.aig.and_inv %A3, %B1, %C1 : i1
  %n19 = synth.aig.and_inv %A3, %B1, %C2 : i1
  %n20 = synth.aig.and_inv %A3, %B1, %C3 : i1
  %n21 = synth.aig.and_inv %A3, %B2, %C1 : i1
  %n22 = synth.aig.and_inv %A3, %B2, %C2 : i1
  %n23 = synth.aig.and_inv %A3, %B2, %C3 : i1
  %n24 = synth.aig.and_inv %A3, %B3, %C1 : i1
  %n25 = synth.aig.and_inv %A3, %B3, %C2 : i1
  %n26 = synth.aig.and_inv %A3, %B3, %C3 : i1
  %n27 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5, not %n6, not %n7, not %n8, not %n9, not %n10, not %n11, not %n12, not %n13, not %n14, not %n15, not %n16, not %n17, not %n18, not %n19, not %n20, not %n21, not %n22, not %n23, not %n24, not %n25, not %n26 : i1
  %n28 = synth.aig.and_inv not %n27 : i1
  hw.output %n28 : i1
}
)mlir"
