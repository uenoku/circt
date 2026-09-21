// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-policy=booth}, hw.module(convert-datapath-to-comb))' | FileCheck %s
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-policy=compress-to-add}, hw.module(convert-datapath-to-comb))' | FileCheck %s --check-prefix=TOADD

// Per-region lowering override: the probe-fed multiplier gets the selected
// policy (Booth partial products here) while the relaxed multiplier keeps
// the heuristic lowering. The relaxed cone matches the all-heuristic
// profile (68 comb.and) and the selected cone the Booth profile (fewer
// AND rows, more replicate/extract glue); the mixture lands between the
// two whole-design profiles.
// CHECK-COUNT-64: comb.and
// TOADD: comb.add bin %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : i8
// TOADD-COUNT-2: comb.add
hw.module @two(in %a : i8, in %b : i8, in %c : i8, in %d : i8, out x : i8, out y : i8) {
  %m0 = comb.mul %a, %b : i8
  %p = synth.timing_probe %m0 : i8 {required_time = 10000 : i64}
  %m1 = comb.mul %c, %d : i8
  hw.output %p, %m1 : i8, i8
}
