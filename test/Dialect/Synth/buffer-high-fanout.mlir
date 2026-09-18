// RUN: circt-opt --pass-pipeline='builtin.module(synth-buffer-high-fanout{max-fanout=2})' %s | FileCheck %s

// Smallest-area identity cell wins as the buffer (area 1.0 over 5.0).
hw.module @buf1(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64}>} {
    hw.output %a : i1
}
hw.module @buf5(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 5.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64}>} {
    hw.output %a : i1
}
// Real inverter: not an identity, never selected despite lower area.
hw.module @inv05(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 0.5 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<negative>>], input_caps = {a = 1.0 : f64}>} {
    %t = hw.constant true
    %0 = comb.xor %a, %t : i1
    hw.output %0 : i1
}
// Flop-like cell with a clock pin (body never simulated; only ports matter).
hw.module @dff(in %d : i1, in %clk : i1 {synth.liberty.pin = {clock = "true"}}, out q : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {d = 1.0 : f64, clk = 1.0 : f64}>} {
    %f = hw.constant false
    hw.output %f : i1
}

// CHECK-LABEL: hw.module @top
hw.module @top(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, in %f : i1, in %g : i1, in %h : i1, in %i : i1, in %j : i1, in %k : i1, in %clk : i1, out x0 : i1, out x1 : i1, out x2 : i1, out x3 : i1, out x4 : i1, out z0 : i1, out z1 : i1, out z2 : i1, out w0 : i1, out w1 : i1) {
    // %x fans out to 5 data loads with max-fanout=2: chunks 2+2+1.
    // CHECK-COUNT-3: hw.instance "buffered" @buf1
    %x = synth.aig.and_inv %a, %b : i1
    %a0 = synth.aig.and_inv %x, %g : i1
    %a1 = synth.aig.and_inv %x, %h : i1
    %a2 = synth.aig.and_inv %x, %i : i1
    %a3 = synth.aig.and_inv %x, %j : i1
    %a4 = synth.aig.and_inv %x, %k : i1
    // %z fans out to 3 clock pins only: untouched (still drives CLK directly).
    // CHECK: %[[Z:.+]] = synth.aig.and_inv %c, %d
    // CHECK-COUNT-3: @dff(d: %{{[a-z0-9_.]+}}: i1, clk: %[[Z]]: i1)
    %z = synth.aig.and_inv %c, %d : i1
    %q0 = hw.instance "f0" @dff(d: %g: i1, clk: %z: i1) -> (q: i1)
    %q1 = hw.instance "f1" @dff(d: %h: i1, clk: %z: i1) -> (q: i1)
    %q2 = hw.instance "f2" @dff(d: %i: i1, clk: %z: i1) -> (q: i1)
    // %w has exactly max-fanout data loads: untouched.
    // CHECK: %[[W:.+]] = synth.aig.and_inv %e, %f
    // CHECK-COUNT-2: synth.aig.and_inv %[[W]],
    %w = synth.aig.and_inv %e, %f : i1
    %w0 = synth.aig.and_inv %w, %j : i1
    %w1 = synth.aig.and_inv %w, %k : i1
    hw.output %a0, %a1, %a2, %a3, %a4, %q0, %q1, %q2, %w0, %w1 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
