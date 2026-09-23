// RUN: circt-opt --pass-pipeline='builtin.module(synth-buffer-high-fanout{max-fanout=2 sta-violators=%S/Inputs/buffer-violators.json})' %s | FileCheck %s
// RUN: circt-opt --pass-pipeline='builtin.module(synth-buffer-high-fanout{max-fanout=2 sta-violators=%S/Inputs/buffer-violators.json})' %s --mlir-pass-statistics -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS

// Timing-guided mode: only STA violator launch nets are buffered (the
// bogus entry is skipped without failing the pass).
hw.module @buf1(in %a : i1, out z : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64}>} {
    hw.output %a : i1
}
hw.module @dff(in %D : i1, in %clk : i1 {synth.liberty.pin = {clock = "true"}}, out Q : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 6.0 : f64, arcs = [#synth.linear_timing_arc<10, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<10, 0, #synth.polarity<positive>>], input_caps = {d = 1.0 : f64, clk = 1.0 : f64}>} {
    %f = hw.constant false
    hw.output %f : i1
}

// CHECK-LABEL: hw.module @sub
hw.module @sub(in %clk : i1, in %a : i1, in %e2 : i1, out y0 : i1, out y1 : i1, out y2 : i1, out t0 : i1, out t1 : i1, out t2 : i1, out z3 : i1) {
    // Violator launch gen/st (flop Q): 3 data loads, max-fanout=2 -> 2 bufs.
    // CHECK: hw.instance "st" @dff
    // CHECK-COUNT-2: hw.instance "buffered_{{.+}}" @buf1
    %s = hw.instance "st" @dff(D: %a: i1, clk: %clk: i1) -> (Q: i1)
    %a0 = synth.aig.and_inv %s, %a : i1
    %a1 = synth.aig.and_inv %s, %a : i1
    %a2 = synth.aig.and_inv %s, %a : i1
    // Non-violator %t with 3 loads: untouched.
    // CHECK: %[[T:.+]] = synth.aig.and_inv %a, %clk
    // CHECK-COUNT-3: synth.aig.and_inv %[[T]],
    %t = synth.aig.and_inv %a, %clk : i1
    %t0 = synth.aig.and_inv %t, %a : i1
    %t1 = synth.aig.and_inv %t, %a : i1
    %t2 = synth.aig.and_inv %t, %a : i1
    // STA endpoint gen/f3 gets a probe (required 9.9ns, then erased);
    // its driver %e2 has fanout 1 (untouched). Bogus endpoints are
    // skipped without failing.
    // STATS-DAG: 1 endpoints-probed
    // STATS-DAG: 2 endpoints-unresolved
    // STATS-DAG: 1 probes-erased
    // STATS-DAG: 2 violators-resolved
    %qq3 = hw.instance "f3" @dff(D: %e2: i1, clk: %clk: i1) -> (Q: i1)
    hw.output %a0, %a1, %a2, %t0, %t1, %t2, %qq3 : i1, i1, i1, i1, i1, i1, i1
}
hw.module @top(in %clk : i1, in %a : i1, in %e2 : i1, out y0 : i1, out y1 : i1, out y2 : i1, out t0 : i1, out t1 : i1, out t2 : i1, out z3 : i1) {
    %y0, %y1, %y2, %t0, %t1, %t2, %z3 = hw.instance "gen" @sub(clk: %clk: i1, a: %a: i1, e2: %e2: i1) -> (y0: i1, y1: i1, y2: i1, t0: i1, t1: i1, t2: i1, z3: i1)
    hw.output %y0, %y1, %y2, %t0, %t1, %t2, %z3 : i1, i1, i1, i1, i1, i1, i1
}
