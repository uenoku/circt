// RUN: circt-opt %s --synth-lower-variadic | FileCheck %s
// RUN: circt-opt %s --synth-lower-variadic=timing-aware=false | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, out f: i2) {
  // CHECK:      %[[RES0:.+]] = synth.aig.and_inv not %a, %b : i2
  // CHECK-NEXT: %[[RES1:.+]] = synth.aig.and_inv %c, not %d : i2
  // CHECK-NEXT: %[[RES2:.+]] = synth.aig.and_inv %e, %[[RES0]] : i2
  // CHECK-NEXT: %[[RES3:.+]] = synth.aig.and_inv %[[RES1]], %[[RES2]] : i2
  %0 = synth.aig.and_inv not %a, %b, %c, not %d, %e : i2
  hw.output %0 : i2
}

// CHECK: hw.module @Add
hw.module @AddMul(in %x: i4, in %y: i4, in %z: i4, out out: i4) {
  // (x + y) * z * constant
  // => (x + y) * (z * constant)
  // CHECK-NEXT: %c5_i4 = hw.constant 5 : i4
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %x, %y : i4
  // CHECK-NEXT: %[[MUL:.+]] = comb.mul %c5_i4, %z : i4
  // CHECK-NEXT: %[[RES:.+]] = comb.mul %[[ADD]], %[[MUL]] : i4
  // CHECK-NEXT: hw.output %[[RES]] : i4
  %c_i5 = hw.constant 5 : i4
  %add = comb.add %x, %y : i4
  %0 = comb.mul %c_i5, %add, %z : i4
  hw.output %0 : i4
}
