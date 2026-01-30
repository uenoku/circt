// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(synth-print-longest-path-analysis{output-file="-" show-top-k-percent=100})' | FileCheck %s
// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(synth-print-longest-path-analysis{output-file="-" show-top-k-percent=100 emit-json=true})' | FileCheck %s --check-prefix JSON

// CHECK:      # Longest Path Analysis result for "parent"
// JSON:               [{"module_name":"parent",

hw.module private @child(in %a : i1, in %b : i1, out x : i1) {
  %r = synth.aig.and_inv %a, %b {sv.namehint = "r"} : i1 // r[0] := max(a[0], b[0]) + 1 = 1
  hw.output %r : i1
}

hw.module private @parent(in %a : i1, in %b : i1, out x : i1, out y : i1) {
  %0 = hw.instance "c1" @child(a: %a: i1, b: %b: i1) -> (x: i1)
  %1 = hw.instance "c2" @child(a: %0: i1, b: %b: i1) -> (x: i1)
  hw.output %0, %1 : i1, i1
}
