// RUN: circt-synth --liberty %S/Inputs/sample.lib %s | FileCheck %s

// A flop maps to the liberty flop cell and combinational logic maps through
// derived costs to the liberty AND cell, all inside circt-synth.

// CHECK: hw.instance {{.*}} @myand2(A: %a: i1, B: %b: i1) -> (X: i1)
// CHECK: hw.instance {{.*}} @mydff(D: %{{.*}}: i1, CLK: %clk: i1) -> (Q: i1)
// CHECK-NOT: seq.firreg
// CHECK-NOT: comb.and
hw.module @top(in %clk : i1, in %a : i1, in %b : i1, out q : i1) {
  %c = seq.to_clock %clk
  %ab = comb.and %a, %b : i1
  %r = seq.firreg %ab clock %c : i1
  hw.output %r : i1
}
