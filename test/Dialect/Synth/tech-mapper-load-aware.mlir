// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area test=true max-cuts-per-root=8 load-aware-delay=true})' %s | FileCheck %s --check-prefixes CHECK,AREA-ON
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=timing test=true max-cuts-per-root=8 load-aware-delay=true})' %s | FileCheck %s --check-prefixes CHECK,TIMING-ON
// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=timing test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefixes CHECK,OFF

// Load-aware selection (Stage 1): both cells implement AND. @and_fast is
// smaller and faster unloaded (intrinsic 100, sensitivity 30) while
// @and_robust is bigger but load-flat (intrinsic 150, sensitivity 0).
// Library mean input cap is 1.0; the shared AND root fans out to 3
// outputs, so output_load = 3 and the load-aware delays are 100+30*3 =
// 190 vs 150 (penalties stay below intrinsic, so no capping binds).
// Area still picks @and_fast (area first); timing picks @and_robust
// (delay first). The area-run arrival of 190 (not the intrinsic 100)
// proves sensitivity is applied at match time.
hw.module @and_fast(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 1.0 : f64, arcs = [#synth.linear_timing_arc<100, 30, #synth.polarity<positive>>, #synth.linear_timing_arc<100, 30, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64, b = 1.0 : f64}>} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_robust(in %a : i1, in %b : i1, out result : i1) attributes {synth.mapping_cost = #synth.mapping_cost<area = 2.0 : f64, arcs = [#synth.linear_timing_arc<150, 0, #synth.polarity<positive>>, #synth.linear_timing_arc<150, 0, #synth.polarity<positive>>], input_caps = {a = 1.0 : f64, b = 1.0 : f64}>} {
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// CHECK-LABEL: hw.module @load_select
// Load-aware ON: area still picks @and_fast (area first) with the
// load-adjusted arrival 250 (not the intrinsic 100); timing picks
// @and_robust (200 < 250).
// Load-aware OFF (default): intrinsics rule; both strategies pick
// @and_fast with arrival 100.
hw.module @load_select(in %a : i1, in %b : i1, out x : i1, out y : i1, out z : i1) {
    // AREA-ON: hw.instance "{{.+}}" @and_fast({{.+}}) {test.arrival_times = [190]}
    // TIMING-ON: hw.instance "{{.+}}" @and_robust({{.+}}) {test.arrival_times = [150]}
    // OFF: hw.instance "{{.+}}" @and_fast({{.+}}) {test.arrival_times = [100]}
    %0 = synth.aig.and_inv %a, %b : i1
    hw.output %0, %0, %0 : i1, i1, i1
}
