// RUN: circt-opt %s --pass-pipeline="builtin.module(hw.module(convert-comb-to-aig{keep-bitwise-logic=true}, cse))" | FileCheck %s

// CHECK-LABEL: @add
hw.module @add(in %lhs: i2, in %rhs: i2, out out: i2) {
  // CHECK:      %[[lhs0:.*]] = comb.extract %lhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[lhs1:.*]] = comb.extract %lhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[rhs0:.*]] = comb.extract %rhs from 0 : (i2) -> i1
  // CHECK-NEXT: %[[rhs1:.*]] = comb.extract %rhs from 1 : (i2) -> i1
  // CHECK-NEXT: %[[sum0:.*]] = comb.xor bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[carry0:.*]] = comb.and bin %[[lhs0]], %[[rhs0]] : i1
  // CHECK-NEXT: %[[sum1:.*]] = comb.xor bin %[[lhs1]], %[[rhs1]], %[[carry0]] : i1
  // CHECK-NEXT: %[[concat:.*]] = comb.concat %[[sum1]], %[[sum0]] : i1, i1
  // CHECK-NEXT: hw.output %[[concat]] : i2
  %0 = comb.add %lhs, %rhs : i2
  hw.output %0 : i2
}

// CHECK-LABEL: @icmp
hw.module @icmp(in %lhs: i2, in %rhs: i2, out out_eq: i1, out out_ne: i1,
                out out_le: i1, out out_lt: i1, out out_ge: i1, out out_gt: i1) {
  %0 = comb.icmp eq %lhs, %rhs : i2
  %1 = comb.icmp ne %lhs, %rhs : i2
  %2 = comb.icmp ule %lhs, %rhs : i2
  %3 = comb.icmp ult %lhs, %rhs : i2
  %4 = comb.icmp uge %lhs, %rhs : i2
  %5 = comb.icmp ugt %lhs, %rhs : i2

  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: @shl
hw.module @shl(in %lhs: i4, in %rhs: i4, out out: i4) {
  %0 = comb.shl %lhs, %rhs : i4
  hw.output %0 : i4
}

hw.module @shl5(in %lhs: i5, in %rhs: i5, out out: i5) {
  %0 = comb.shl %lhs, %rhs : i5
  hw.output %0 : i5
}
