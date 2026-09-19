// RUN: circt-opt --pass-pipeline='builtin.module(synth-buffer-high-fanout{max-fanout=1})' %s | FileCheck %s

// Buffer cell (intrinsic 10) and logic cell (intrinsic 100, flat).
hw.module @buf1(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64}>} {
    hw.output %a : i1
}
hw.module @and100(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<100, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64, b = 1.0 : f64}>} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}
hw.module @dff(in %D : i1, in %CLK : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {D = 1.0 : f64, CLK = 1.0 : f64}>} {
    %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
    hw.output %q : i1
}

// CHECK-LABEL: hw.module @top
hw.module @top(in %clk : i1, in %a : i1, in %b0 : i1, in %b1 : i1, in %b2 : i1, in %c0 : i1, in %c1 : i1, in %c2 : i1, out y : i1, out z : i1, out w0 : i1, out w1 : i1, out w2 : i1) {
    %q = hw.instance "fq" @dff(D: %a: i1, CLK: %clk: i1) -> (Q: i1)
    // Violated probe: arrival 200 (Q=0 + 100 + 100) over required 150.
    // Launch %q (fanout 2 > max-fanout=1) gets chunked: 2 buffers.
    // (Cone args have fanout 1 each: untouched.)
    // CHECK-COUNT-2: hw.instance "buffered" @buf1
    %g0 = hw.instance "g0" @and100(a: %q: i1, b: %b0: i1) -> (result: i1)
    %g1 = hw.instance "g1" @and100(a: %g0: i1, b: %b1: i1) -> (result: i1)
    %p1 = synth.timing_probe %g1 : i1 {required_time = 150 : i64}
    // Satisfied probe: arrival 100 under required 500. Its cone launches
    // nothing new (only %q and fanout-1 args).
    %g2 = hw.instance "g2" @and100(a: %q: i1, b: %b2: i1) -> (result: i1)
    %p2 = synth.timing_probe %g2 : i1 {required_time = 500 : i64}
    // Non-probe net %t with 3 loads: skipped (guided mode gates by launches).
    // CHECK: %[[T:.+]] = synth.aig.and_inv %a, %c0
    // CHECK-COUNT-3: @and100(a: %[[T]]:
    %t = synth.aig.and_inv %a, %c0 : i1
    %w0 = hw.instance "h0" @and100(a: %t: i1, b: %c0: i1) -> (result: i1)
    %w1 = hw.instance "h1" @and100(a: %t: i1, b: %c1: i1) -> (result: i1)
    %w2 = hw.instance "h2" @and100(a: %t: i1, b: %c2: i1) -> (result: i1)
    hw.output %p1, %p2, %w0, %w1, %w2 : i1, i1, i1, i1, i1
}
