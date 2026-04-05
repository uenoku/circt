// RUN: circt-opt %s --synth-test-priority-cuts='max-cut-input-size=3 max-cuts-per-root=8' | FileCheck %s

// This test exercises choice cuts whose alternative operands expose different
// internal leaves. Truth-table computation for the copied choice cuts must use
// the selected operand cut's leaves, not resimulate through the first choice
// operand.
//
// CHECK-LABEL: Enumerating cuts for module: choice_expanded_truth_table
// CHECK: choice 4 cuts: {choice}@t2d0 {a, b, c}@t128d1 {a, bc}@t8d2 {c, ab}@t8d2
// CHECK: out 4 cuts: {out}@t2d0 {d, choice}@t8d2 {a, d, bc}@t128d2 {c, d, ab}@t128d2
hw.module @choice_expanded_truth_table(in %a : i1, in %b : i1, in %c : i1,
                                       in %d : i1, out y : i1) {
  %ab = synth.aig.and_inv %a, %b {sv.namehint = "ab"} : i1
  %abc_left = synth.aig.and_inv %ab, %c {sv.namehint = "abc_left"} : i1
  %bc = synth.aig.and_inv %b, %c {sv.namehint = "bc"} : i1
  %abc_right = synth.aig.and_inv %a, %bc {sv.namehint = "abc_right"} : i1
  %choice = synth.choice %abc_left, %abc_right {sv.namehint = "choice"} : i1
  %out = synth.aig.and_inv %choice, %d {sv.namehint = "out"} : i1
  hw.output %out : i1
}
