// REQUIRES: opentimer
// RUN: circt-synth %s --top=top --liberty=%S/Inputs/tiny-timing.lib --datapath-retry=1 --retry-timeout=120 --timing-backend=opentimer --timing-clock-period-ns=0.5 --timing-clock-port=clk -o %t.mlir 2>&1 | FileCheck %s
// The in-process OpenTimer backend judges candidates without Verilog
// export, SDC parsing, or a `sta` executable.
// CHECK: baseline: inst={{.*}} slack=
// CHECK: BEST

hw.module @top(in %a : i1, in %clk : i1, out q : i1) {
  %ck = seq.to_clock %clk
  %y = comb.and %a, %a : i1
  %q = seq.compreg %y, %ck : i1
  hw.output %q : i1
}
