// RUN: circt-opt --synth-dedup-instances %s | FileCheck %s

hw.module.extern @inv(in %a: i1, out y: i1)
hw.module.extern @buf(in %a: i1, out y: i1)
hw.module.extern @dff(in %clk: !seq.clock, in %d: i1, out q: i1)
// i1-clocked flop cell with liberty clock metadata.
hw.module.extern @sdff_lib(in %clk: i1 {synth.liberty.pin = {clock = "true"}}, in %d: i1, out q: i1)

// CHECK-LABEL: hw.module @top
hw.module @top(in %a: i1, in %clk: !seq.clock, out x: i1, out y: i1, out z: i1, out w: i1) {
  // i2 duplicates i1 and is merged away.
  // CHECK: hw.instance "i1" @inv
  // CHECK-NOT: hw.instance "i2" @inv
  // CHECK: hw.instance "i3" @inv
  %y1 = hw.instance "i1" @inv(a: %a: i1) -> (y: i1)
  %y2 = hw.instance "i2" @inv(a: %a: i1) -> (y: i1)
  %y3 = hw.instance "i3" @inv(a: %y1: i1) -> (y: i1)
  // Same input but a different cell is kept.
  // CHECK: hw.instance "b" @buf
  %b = hw.instance "b" @buf(a: %a: i1) -> (y: i1)
  // Flops with identical inputs are NOT merged (clock operand).
  // CHECK: hw.instance "f1" @dff
  // CHECK: hw.instance "f2" @dff
  %q1 = hw.instance "f1" @dff(clk: %clk: !seq.clock, d: %a: i1) -> (q: i1)
  %q2 = hw.instance "f2" @dff(clk: %clk: !seq.clock, d: %a: i1) -> (q: i1)
  // i1-clocked flop duplicates are NOT merged (liberty clock pin).
  // CHECK: hw.instance "g1" @sdff_lib
  // CHECK: hw.instance "g2" @sdff_lib
  %r1 = hw.instance "g1" @sdff_lib(clk: %a: i1, d: %a: i1) -> (q: i1)
  %r2 = hw.instance "g2" @sdff_lib(clk: %a: i1, d: %a: i1) -> (q: i1)
  // Hierarchical duplicates are NOT merged (leaf cells only).
  // CHECK: hw.instance "h1" @sub
  // CHECK: hw.instance "h2" @sub
  %s1 = hw.instance "h1" @sub(a: %a: i1) -> (y: i1)
  %s2 = hw.instance "h2" @sub(a: %a: i1) -> (y: i1)
  // CHECK: hw.output
  hw.output %y2, %y3, %q1, %r1 : i1, i1, i1, i1
  // CHECK-NOT: "i2"
}

// Submodule with a body: instances of it never merge (leaf rule).
hw.module @sub(in %a: i1, out y: i1) {
  %y = hw.instance "inner" @inv(a: %a: i1) -> (y: i1)
  hw.output %y : i1
}
