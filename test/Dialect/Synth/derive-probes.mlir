// RUN: circt-opt --pass-pipeline='builtin.module(synth-derive-probes{probe-clocks=clk probe-period-ns=10.0})' %s | FileCheck %s
// RUN: circt-opt --pass-pipeline='builtin.module(synth-derive-probes{probe-clocks=clk probe-period-ns=10.0},synth-derive-probes{probe-clocks=clk probe-period-ns=10.0})' %s | FileCheck %s --check-prefix=IDEMPOTENT

// Buffer cell (untouched by derive; presence keeps the file realistic).
hw.module @buf1(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64}>} {
    hw.output %a : i1
}
// Flop cell with clock pin (body has synth.ff so it counts as a flop).
hw.module @fd1(in %D : i1, in %CLK : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.liberty.cell = {area = 6.0 : f64}} {
    %q, %qn = synth.ff %D clock %CLK edge rise : i1, i1
    hw.output %q : i1
}

// CHECK-LABEL: hw.module @sub
hw.module @sub(in %clk : i1, in %d : i1, in %e : i1, out q : i1, out r : i1) {
    // Clocked flop D gets a probe with required_time = 10ns = 10000ps.
    // CHECK: %[[P:.+]] = synth.timing_probe %d : i1 {required_time = 10000 : i64}
    // CHECK: hw.instance "f0" @fd1(D: %[[P]]: i1, CLK: %clk: i1)
    %q0 = hw.instance "f0" @fd1(D: %d: i1, CLK: %clk: i1) -> (Q: i1)
    // Unclocked flop (data-driven clock): no probe, D untouched.
    // CHECK: hw.instance "f1" @fd1(D: %e: i1, CLK: %d: i1)
    %q1 = hw.instance "f1" @fd1(D: %e: i1, CLK: %d: i1) -> (Q: i1)
    hw.output %q0, %q1 : i1, i1
}
// CHECK-LABEL: hw.module @top
hw.module @top(in %clk : i1, in %d : i1, in %e : i1, out q : i1, out r : i1) {
    %q, %r = hw.instance "gen" @sub(clk: %clk: i1, d: %d: i1, e: %e: i1) -> (q: i1, r: i1)
    hw.output %q, %r : i1, i1
}

// Repeated derivation never nests probes: the second run sees D already
// driven by a probe and skips the endpoint, so @sub keeps exactly one.
// IDEMPOTENT-LABEL: hw.module @sub
// IDEMPOTENT-COUNT-1: synth.timing_probe
