// RUN: mkdir -p %t && circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-report=%t/a.json})' -o /dev/null && circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-datapath, canonicalize, cse), synth-discover-refinement-regions{refinement-report=%t/b.json})' -o /dev/null && diff %t/a.json %t/b.json

// Determinism: identical refinement reports across repeated runs (stable
// region IDs, ordering, and statistics, excluding elapsed time).
hw.module @arith(in %a : i8, in %b : i8, in %c : i8, in %d : i8, in %e : i8, out x : i8, out y : i8) {
  %add = comb.add %a, %b, %c : i8
  %p1 = synth.timing_probe %add : i8 {required_time = 10000 : i64}
  %mul = comb.mul %d, %e : i8
  %p2 = synth.timing_probe %mul : i8 {required_time = 10000 : i64}
  hw.output %p1, %p2 : i8, i8
}
