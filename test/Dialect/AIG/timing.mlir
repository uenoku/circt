// RUN: circt-opt %s --split-input-file --pass-pipeline='builtin.module(aig-print-longest-paths-analysis{test=true})' --verify-diagnostics

// expected-remark @below {{OpenPath(fanOut=x[0], fanIn=Object($root.a[0], delay=1, history=[{{.+}}]))}}
// expected-remark @below {{OpenPath(fanOut=x[0], fanIn=Object($root.b[0], delay=1, history=[{{.+}}]))}}
// expected-remark @below {{OpenPath(fanOut=x[1], fanIn=Object($root.a[1], delay=1, history=[{{.+}}]))}}
// expected-remark @below {{OpenPath(fanOut=x[1], fanIn=Object($root.b[1], delay=1, history=[{{.+}}]))}}
hw.module private @basicWord(in %a : i2, in %b : i2, out x : i2) {
  %r = aig.and_inv not %a, %b : i2
  hw.output %r : i2
}

// -----

// expected-remark @+1 {{OpenPath(fanOut=x[0], fanIn=Object($root.a[0], delay=2, history=[{{.+}})]))}}
hw.module private @max(in %a : i1, out x : i1) {
  %r = aig.and_inv not %a : i1 // r[0] := delay 1
  %q = aig.and_inv %r, %a : i1 // q[0] := max(r[0], a[0]) + 1 = 2
  hw.output %q : i1
}


// // expected-remark @+1 {{OpenPath(fanOut=x[0], fanIn=Object($root.a[0], delay=2, history=[{{.+}})]))}}
// hw.module private @extract(in %a : i3, out x : i1) {
//   %0 = comb.extract %a from 0 : (i3) -> i1
//   %1 = comb.extract %a from 1 : (i3) -> i1
//   %2 = comb.extract %a from 2 : (i3) -> i1
//   %q = aig.and_inv %0, %1 : i1 // q[0] := max(r[0], a[0]) + 1 = 2
//   %r = aig.and_inv %q, %2 : i1 // r[0] := max(q[0], a[2]) + 1 = 3
//   hw.output %r : i1
// }


// hw.module private @basic(in %clock : !seq.clock, in %a : i2, in %b : i2, out x : i2, out y : i2) {
//   %p = seq.firreg %a clock %clock : i2
//   %q = seq.firreg %b clock %clock : i2
//   %r = aig.and_inv not %p, %q : i2
//   %u = comb.and %r, %q : i2
//   %v = comb.or %u, %r : i2
//   %out = seq.firreg %v clock %clock : i2
//   hw.output %r : i2
// }

// hw.module private @xor(in %clock : !seq.clock, in %arg0 : i2, in %arg1 : i32, in %arg2 : i32, out out0 : i32) {
//   %2 = aig.and_inv not %0, not %1 : i2
//   hw.output %foo : i2
// }
// 
// hw.module @top(in %clock : !seq.clock, in %arg0 : i2, in %arg1 + startIndex : i32, in %arg2 : i32, out out0 : i32) {
//   %0 = aig.and_inv %arg0, %arg1 : i2
//   %1 = aig.and_inv %arg1, %arg2 : i2
//   %2 = hw.instance "a1" @xor(clock: %clock: !seq.clock, arg0: %0: i2, arg1: %1: i32, arg2: %0: i32) -> (out0: i32)
//   hw.output %2 : i2
// }
