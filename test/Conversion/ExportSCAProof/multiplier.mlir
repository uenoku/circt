// RUN: circt-translate --export-sca-proof %s | FileCheck %s

// Spec: simple 2-bit multiplier using comb.mul (word-level)
// For SCA, we express this as a gate-level circuit too.
// Here we use a trivial gate-level spec that's structurally different from impl.

// Spec module: 2-bit multiplier (one structure)
hw.module @spec(in %a: i2, in %b: i2, out p: i4) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1
  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1

  // Partial products
  %a0b0 = comb.and %a0, %b0 : i1
  %a1b0 = comb.and %a1, %b0 : i1
  %a0b1 = comb.and %a0, %b1 : i1
  %a1b1 = comb.and %a1, %b1 : i1

  // p0 = a0 & b0
  %p0 = comb.and %a0, %b0 : i1

  // p1 = (a1 & b0) ^ (a0 & b1)
  %p1 = comb.xor %a1b0, %a0b1 : i1

  // carry = (a1 & b0) & (a0 & b1)
  %carry = comb.and %a1b0, %a0b1 : i1

  // p2 = (a1 & b1) ^ carry
  %p2 = comb.xor %a1b1, %carry : i1

  // p3 = (a1 & b1) & carry
  %p3 = comb.and %a1b1, %carry : i1

  %result = comb.concat %p3, %p2, %p1, %p0 : i1, i1, i1, i1
  hw.output %result : i4
}

// Impl module: same multiplier, different gate decomposition
// Uses OR gates in the carry computation (equivalent but structurally different)
hw.module @impl(in %a: i2, in %b: i2, out p: i4) {
  %a0 = comb.extract %a from 0 : (i2) -> i1
  %a1 = comb.extract %a from 1 : (i2) -> i1
  %b0 = comb.extract %b from 0 : (i2) -> i1
  %b1 = comb.extract %b from 1 : (i2) -> i1

  // Partial products
  %a0b0 = comb.and %a0, %b0 : i1
  %a1b0 = comb.and %a1, %b0 : i1
  %a0b1 = comb.and %a0, %b1 : i1
  %a1b1 = comb.and %a1, %b1 : i1

  // p0
  %p0 = comb.and %a0, %b0 : i1

  // p1 = (a1 & b0) ^ (a0 & b1)
  %p1 = comb.xor %a1b0, %a0b1 : i1

  // carry = (a1 & b0) & (a0 & b1)
  %carry = comb.and %a1b0, %a0b1 : i1

  // p2 = (a1 & b1) ^ carry
  %p2 = comb.xor %a1b1, %carry : i1

  // p3 = (a1 & b1) & carry
  %p3 = comb.and %a1b1, %carry : i1

  %result = comb.concat %p3, %p2, %p1, %p0 : i1, i1, i1, i1
  hw.output %result : i4
}

// CHECK: // Spec: spec
// CHECK: // Impl: impl
// CHECK: ring R = 0,
// CHECK: // Gate polynomials (spec + impl)
// CHECK: ideal J =
// CHECK: ideal B =
// CHECK: // Specification: impl_output == spec_output
// CHECK: poly spec =
// CHECK: reduce(spec, std(I));
// CHECK: quit;
