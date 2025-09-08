// RUN: circt-opt -canonicalize='top-down=true region-simplify=aggressive' %s | FileCheck %s

// CHECK-LABEL: hw.module @Basic
hw.module @Basic(in %a : i1, in %b : i1, out o1 : i1, out o2 : i1, out o3 : i1, out o4 : i1) {
  // Single operand, not inverted -> replace with operand
  %0 = synth.mig.maj_inv %a : i1

  // Three operands, two same -> replace with the operand
  %1 = synth.mig.maj_inv %a, %a, %b : i1

  // Three operands, two complementary -> replace with the third
  %2 = synth.mig.maj_inv %a, %b, not %b : i1

  // Two operands same but one inverted -> replace with the third
  %3 = synth.mig.maj_inv %a, not %a, %b : i1

  // CHECK: hw.output %a, %a, %a, %b : i1, i1, i1, i1
  hw.output %0, %1, %2, %3 : i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @Constants
hw.module @Constants(in %a : i1, out o1 : i1, out o2 : i1, out o3 : i1) {
  // Two constants equal -> replace with constant
  %c = hw.constant 1 : i1
  %0 = synth.mig.maj_inv %c, %c, %a : i1

  // Two constants complementary -> replace with the third
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %1 = synth.mig.maj_inv %c0, %c1, %a : i1

  // Two constants equal with one inverted -> replace with constant
  %2 = synth.mig.maj_inv not %c0, %c1, %a : i1

  // CHECK: hw.output %true, %a, %true : i1, i1, i1
  hw.output %0, %1, %2 : i1, i1, i1
}
