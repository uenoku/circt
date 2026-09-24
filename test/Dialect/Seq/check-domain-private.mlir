// RUN: circt-opt --seq-check-domain='module-name=Top' %s | FileCheck %s

// The selected module is private and has a use in the surrounding design. The
// checker must analyze the cloned module while leaving the original hierarchy
// and its hierarchical paths intact.
// CHECK: hw.hierpath private @ChildPath [@Child]
hw.module @Parent(in %clock: !seq.clock, in %in: i8) {
  hw.instance "top" @Top(clock: %clock: !seq.clock, in: %in: i8) -> ()
  hw.output
}

// CHECK-LABEL: hw.module private @Top_flatten
// CHECK-NOT: hw.instance
// CHECK: sv.verbatim "top ref" {symbols = [#hw.innerNameRef<@Top_flatten::@top_wire>]}
// CHECK: hw.wire {{.*}} sym @wire
// CHECK: sv.verbatim "ref" {symbols = [#hw.innerNameRef<@Top_flatten::@wire>]}
// CHECK: seq.check_clock_domain
hw.module private @Top(in %clock: !seq.clock, in %in: i8) {
  %top_wire = hw.wire %in sym @top_wire : i8
  sv.verbatim "top ref" {symbols = [#hw.innerNameRef<@Top::@top_wire>]}
  %value = hw.instance "child" @Child(clock: %clock: !seq.clock, in: %top_wire: i8) -> (out: i8)
  seq.check_clock_domain %value, %clock : i8, !seq.clock
  hw.output
}

hw.hierpath private @ChildPath [@Child]

hw.module private @Child(in %clock: !seq.clock, in %in: i8, out out: i8) {
  %reg = seq.compreg %in, %clock : i8
  %wire = hw.wire %reg sym @wire : i8
  sv.verbatim "ref" {symbols = [#hw.innerNameRef<@Child::@wire>]}
  hw.output %wire : i8
}
