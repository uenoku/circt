R"mlir(
// Generated from mockturtle experiments/cell_libraries/sky130.genlib.
// Constants are omitted; synthesis keeps constants in IR.
hw.module @sky130_fd_sc_hd__buf_2(in %A : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 5.0 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 13001, 0, #synth.polarity<positive>>], input_caps = {}>} {
  hw.output %A : i1
}

hw.module @sky130_fd_sc_hd__bufinv_8(in %A : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 17.52 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 7276, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__or2_2(in %A : i1, in %B : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 17360, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 16185, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__and2_2(in %A : i1, in %B : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 7.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 12817, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 13992, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__and2b_2(in %A_N : i1, in %B : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A_N", 13446, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 13174, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, %B : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__or2b_2(in %A : i1, in %B_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 14605, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B_N", 13522, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, %B_N : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__nor2_2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.26 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 7758, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 6172, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__nand2_2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.26 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 5107, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 5999, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__xnor2_2(in %A : i1, in %B : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 17116, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 15139, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B : i1
  %n1 = synth.aig.and_inv %A, %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__xor2_2(in %A : i1, in %B : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 18799, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 16040, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, not %B : i1
  %n1 = synth.aig.and_inv not %A, %B : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__and3b_2(in %A_N : i1, in %B : i1, in %C : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A_N", 20949, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 16371, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 17300, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, %B, %C : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__and3_2(in %A : i1, in %B : i1, in %C : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 7.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 14653, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 15947, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 16784, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__or3b_2(in %A : i1, in %B : i1, in %C_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 25185, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 23948, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C_N", 28751, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, %C_N : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__or3_2(in %A : i1, in %B : i1, in %C : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 7.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 25135, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 24003, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 21926, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__a21bo_2(in %A1 : i1, in %A2 : i1, in %B1_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 14576, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 15474, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1_N", 18521, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, %B1_N : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o21ba_2(in %A1 : i1, in %A2 : i1, in %B1_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 16305, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 14619, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1_N", 17292, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, not %B1_N : i1
  %n1 = synth.aig.and_inv %A2, not %B1_N : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__mux2_2(in %A0 : i1, in %A1 : i1, in %S : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A0", 18996, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A1", 19077, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "S", 25712, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A0, not %S : i1
  %n1 = synth.aig.and_inv %A1, %S : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a21o_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 14047, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 15167, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 12385, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o21a_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 15573, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 14127, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 10700, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A2, %B1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__o21bai_2(in %A1 : i1, in %A2 : i1, in %B1_N : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 10385, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 8469, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1_N", 13099, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1_N : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__nor3b_2(in %A : i1, in %B : i1, in %C_N : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 11969, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 10966, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C_N", 14060, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, %C_N : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__nand3b_2(in %A_N : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"Y", "A_N", 13249, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 7253, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 7780, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, %B, %C : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__a21boi_2(in %A1 : i1, in %A2 : i1, in %B1_N : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 8854, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 10217, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1_N", 13589, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, %B1_N : i1
  %n1 = synth.aig.and_inv not %A2, %B1_N : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__nor3_2(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 11545, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 10639, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 7912, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__nand3_2(in %A : i1, in %B : i1, in %C : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 5877, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 7073, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 7471, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__o21ai_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 10236, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 8580, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 5537, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %n0, %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a21oi_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 8713, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 10106, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 7223, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__or4b_2(in %A : i1, in %B : i1, in %C : i1, in %D_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 33644, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 32309, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 30799, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D_N", 33774, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, %D_N : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__or4_2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 33417, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 32504, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 30698, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D", 27173, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, not %D : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__and4bb_2(in %A_N : i1, in %B_N : i1, in %C : i1, in %D : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A_N", 19909, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B_N", 20481, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 19787, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D", 17989, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, not %B_N, %C, %D : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__and4_2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 16110, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 17799, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 18764, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D", 19525, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C, %D : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__and4b_2(in %A_N : i1, in %B : i1, in %C : i1, in %D : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A_N", 22997, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 18406, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C", 19408, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D", 20395, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, %B, %C, %D : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__a2bb2o_2(in %A1_N : i1, in %A2_N : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1_N", 22016, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2_N", 20902, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 17161, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 16013, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %B1, %B2 : i1
  %n1 = synth.aig.and_inv not %A1_N, not %A2_N : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__o2bb2a_2(in %A1_N : i1, in %A2_N : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1_N", 18809, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2_N", 18321, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 17807, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 16393, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1_N, %B1 : i1
  %n1 = synth.aig.and_inv not %A2_N, %B1 : i1
  %n2 = synth.aig.and_inv not %A1_N, %B2 : i1
  %n3 = synth.aig.and_inv not %A2_N, %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o31a_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 22582, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 21398, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 18432, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 11115, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A2, %B1 : i1
  %n2 = synth.aig.and_inv %A3, %B1 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @sky130_fd_sc_hd__a22o_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 16949, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 18160, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 15000, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 15793, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %B1, %B2 : i1
  %n1 = synth.aig.and_inv %A1, %A2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a211o_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 19808, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 21186, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 18729, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 16502, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1, not %C1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a31o_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 8.76 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 15968, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 17339, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 18198, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 13937, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o22a_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 17686, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 16268, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 15393, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 13846, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A2, %B1 : i1
  %n2 = synth.aig.and_inv %A1, %B2 : i1
  %n3 = synth.aig.and_inv %A2, %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o211a_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 10.01 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 19554, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 17994, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 13683, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 12740, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__or4bb_2(in %A : i1, in %B : i1, in %C_N : i1, in %D_N : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A", 23771, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B", 24261, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C_N", 22853, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D_N", 20754, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, %C_N, %D_N : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__nor4b_2(in %A : i1, in %B : i1, in %C : i1, in %D_N : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 16774, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 15932, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 13839, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D_N", 14819, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, %D_N : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__nand4b_2(in %A_N : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A_N", 13768, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 8639, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 9589, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 9926, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A_N, %B, %C, %D : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__a22oi_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 11431, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 13027, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 8589, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 10265, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A1, not %B2 : i1
  %n2 = synth.aig.and_inv not %A2, not %B1 : i1
  %n3 = synth.aig.and_inv not %A2, not %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__nor4_2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 16642, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 15728, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 13717, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 9743, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A, not %B, not %C, not %D : i1
  hw.output %n0 : i1
}

hw.module @sky130_fd_sc_hd__o22ai_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 12451, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 10667, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 9530, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 7773, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %B1, not %B2 : i1
  %n1 = synth.aig.and_inv not %A1, not %A2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a211oi_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 12937, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 15012, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 12926, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 9945, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1, not %C1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1, not %C1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__nand4_2(in %A : i1, in %B : i1, in %C : i1, in %D : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A", 6762, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B", 8532, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C", 9339, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D", 9582, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A, %B, %C, %D : i1
  %n1 = synth.aig.and_inv not %n0 : i1
  hw.output %n1 : i1
}

hw.module @sky130_fd_sc_hd__o211ai_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 12994, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 11208, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 7872, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 6712, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %n0, %B1, %C1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o31ai_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 16209, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 14805, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 12248, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 6151, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2, not %A3 : i1
  %n1 = synth.aig.and_inv not %n0, %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a31oi_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 10389, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 12303, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 13542, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 8993, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1 : i1
  %n2 = synth.aig.and_inv not %A3, not %B1 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @sky130_fd_sc_hd__a2bb2oi_2(in %A1_N : i1, in %A2_N : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1_N", 14773, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2_N", 13229, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 9933, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 8677, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1_N, not %B1 : i1
  %n1 = synth.aig.and_inv %A1_N, not %B2 : i1
  %n2 = synth.aig.and_inv %A2_N, not %B1 : i1
  %n3 = synth.aig.and_inv %A2_N, not %B2 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o2bb2ai_2(in %A1_N : i1, in %A2_N : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1_N", 11617, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2_N", 10869, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 10316, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 8563, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %B1, not %B2 : i1
  %n1 = synth.aig.and_inv %A1_N, %A2_N : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a311o_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 22199, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 25112, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 26627, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 21625, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 18826, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1, not %C1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o32a_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 25134, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 23767, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 21086, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 16042, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 14357, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A1, %B2 : i1
  %n2 = synth.aig.and_inv %A2, %B1 : i1
  %n3 = synth.aig.and_inv %A3, %B1 : i1
  %n4 = synth.aig.and_inv %A2, %B2 : i1
  %n5 = synth.aig.and_inv %A3, %B2 : i1
  %n6 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5 : i1
  %n7 = synth.aig.and_inv not %n6 : i1
  hw.output %n7 : i1
}

hw.module @sky130_fd_sc_hd__a221o_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 23914, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 25858, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 22537, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 24341, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 19275, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %B1, %B2 : i1
  %n1 = synth.aig.and_inv %A1, %A2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1, not %C1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a41o_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %A4 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 17653, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 19877, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 21137, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A4", 21997, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 14899, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3, %A4 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a32o_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 19700, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 21324, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 21754, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 16423, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 17395, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2, %A3 : i1
  %n1 = synth.aig.and_inv %B1, %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__o221a_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 22065, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 20596, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 19629, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B2", 18160, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 13013, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n2 = synth.aig.and_inv %A1, %B2, %C1 : i1
  %n3 = synth.aig.and_inv %A2, %B2, %C1 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o311a_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %C1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 11.26 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 26481, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 25058, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 21900, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 14064, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 12890, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1 : i1
  %n1 = synth.aig.and_inv %A2, %B1, %C1 : i1
  %n2 = synth.aig.and_inv %A3, %B1, %C1 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @sky130_fd_sc_hd__o41a_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %A4 : i1, in %B1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 29671, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 28329, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 26362, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A4", 22345, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 11163, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1 : i1
  %n1 = synth.aig.and_inv %A2, %B1 : i1
  %n2 = synth.aig.and_inv %A3, %B1 : i1
  %n3 = synth.aig.and_inv %A4, %B1 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o2111a_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, in %D1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 22797, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 20679, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 16501, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 15617, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D1", 14120, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %B1, %C1, %D1 : i1
  %n1 = synth.aig.and_inv %A2, %B1, %C1, %D1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a2111o_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, in %D1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 12.51 : f64, arcs = [#synth.linear_timing_arc<"X", "A1", 26744, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 28932, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "B1", 26074, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "C1", 24043, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "D1", 20293, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A1, %A2 : i1
  %n1 = synth.aig.and_inv not %n0, not %B1, not %C1, not %D1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a32oi_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 13177, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 15423, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 16276, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 10150, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 11675, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A1, not %B2 : i1
  %n2 = synth.aig.and_inv not %A2, not %B1 : i1
  %n3 = synth.aig.and_inv not %A3, not %B1 : i1
  %n4 = synth.aig.and_inv not %A2, not %B2 : i1
  %n5 = synth.aig.and_inv not %A3, not %B2 : i1
  %n6 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3, not %n4, not %n5 : i1
  %n7 = synth.aig.and_inv not %n6 : i1
  hw.output %n7 : i1
}

hw.module @sky130_fd_sc_hd__o2111ai_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, in %D1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 16171, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 14070, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 10633, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 9736, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D1", 7841, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2 : i1
  %n1 = synth.aig.and_inv not %n0, %B1, %C1, %D1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__o41ai_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %A4 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 21819, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 20688, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 18624, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A4", 14181, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 5994, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2, not %A3, not %A4 : i1
  %n1 = synth.aig.and_inv not %n0, %B1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a41oi_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %A4 : i1, in %B1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 10977, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 13489, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 15103, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A4", 16114, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 9896, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1 : i1
  %n2 = synth.aig.and_inv not %A3, not %B1 : i1
  %n3 = synth.aig.and_inv not %A4, not %B1 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__a311oi_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 14871, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 17369, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 19126, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 15357, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 12290, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1, not %C1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1, not %C1 : i1
  %n2 = synth.aig.and_inv not %A3, not %B1, not %C1 : i1
  %n3 = synth.aig.and_inv not %n0, not %n1, not %n2 : i1
  %n4 = synth.aig.and_inv not %n3 : i1
  hw.output %n4 : i1
}

hw.module @sky130_fd_sc_hd__o32ai_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %B2 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 18245, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 17183, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 14319, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 10122, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 8258, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2, not %A3 : i1
  %n1 = synth.aig.and_inv not %B1, not %B2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__o311ai_2(in %A1 : i1, in %A2 : i1, in %A3 : i1, in %B1 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 16.27 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 19690, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 18323, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A3", 15466, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 8974, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 7636, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %A2, not %A3 : i1
  %n1 = synth.aig.and_inv not %n0, %B1, %C1 : i1
  %n2 = synth.aig.and_inv not %n1 : i1
  hw.output %n2 : i1
}

hw.module @sky130_fd_sc_hd__a221oi_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 15977, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 17469, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 14212, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 15914, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 11214, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1, not %C1 : i1
  %n1 = synth.aig.and_inv not %A1, not %B2, not %C1 : i1
  %n2 = synth.aig.and_inv not %A2, not %B1, not %C1 : i1
  %n3 = synth.aig.and_inv not %A2, not %B2, not %C1 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}

hw.module @sky130_fd_sc_hd__o221ai_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %B2 : i1, in %C1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 15117, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 13206, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 13277, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B2", 11206, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 7326, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %B1, not %B2 : i1
  %n1 = synth.aig.and_inv not %A1, not %A2 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1, %C1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__a2111oi_2(in %A1 : i1, in %A2 : i1, in %B1 : i1, in %C1 : i1, in %D1 : i1, out Y : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 15.01 : f64, arcs = [#synth.linear_timing_arc<"Y", "A1", 19122, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "A2", 21994, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "B1", 19841, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "C1", 17358, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"Y", "D1", 13060, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv not %A1, not %B1, not %C1, not %D1 : i1
  %n1 = synth.aig.and_inv not %A2, not %B1, not %C1, not %D1 : i1
  %n2 = synth.aig.and_inv not %n0, not %n1 : i1
  %n3 = synth.aig.and_inv not %n2 : i1
  hw.output %n3 : i1
}

hw.module @sky130_fd_sc_hd__mux4_1(in %A0 : i1, in %A1 : i1, in %A2 : i1, in %A3 : i1, in %S0 : i1, in %S1 : i1, out X : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 26.28 : f64, arcs = [#synth.linear_timing_arc<"X", "A0", 31580, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A1", 32198, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A2", 31897, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "A3", 31634, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "S0", 40601, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<"X", "S1", 22077, 0, #synth.polarity<positive>>], input_caps = {}>} {
  %n0 = synth.aig.and_inv %A0, not %S0, not %S1 : i1
  %n1 = synth.aig.and_inv %A1, %S0, not %S1 : i1
  %n2 = synth.aig.and_inv %A2, not %S0, %S1 : i1
  %n3 = synth.aig.and_inv %A3, %S0, %S1 : i1
  %n4 = synth.aig.and_inv not %n0, not %n1, not %n2, not %n3 : i1
  %n5 = synth.aig.and_inv not %n4 : i1
  hw.output %n5 : i1
}
)mlir"
