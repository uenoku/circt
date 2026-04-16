// RUN: circt-opt %s --synth-backward-signal-tracker='target-signals=wire_a top-module-name=simple' | FileCheck %s

// CHECK-LABEL: === Backward Signal Tracker ===
// CHECK: Target signal paths: wire_a
// CHECK: Top module: simple

hw.module @simple(in %a: i1, in %b: i1, out out: i1) {
  %wire_a = comb.and %a, %b {name = "wire_a"} : i1
  %wire_b = comb.or %wire_a, %a {name = "wire_b"} : i1
  %wire_c = comb.xor %wire_b, %wire_a {name = "wire_c"} : i1
  hw.output %wire_c : i1
}

// CHECK: Found {{[0-9]+}} target object(s)
// CHECK: Tracking from:
// CHECK: User:
