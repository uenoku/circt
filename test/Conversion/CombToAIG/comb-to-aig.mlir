// RUN: circt-opt %s --convert-comb-to-aig | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, in %arg4: i1, out out: i32) {
  // CHECK-NEXT: %0 = aig.and_inv not %arg0, not %arg1, not %arg2, not %arg3 : i32
  // CHECK-NEXT: %1 = aig.and_inv not %0 : i32
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @mux
hw.module @mux(in %cond: i1, in %high: i32, in %low: i32, out out: i32) {
  // CHECK-NEXT: %[[REPLICATE0:.*]] = comb.replicate %cond : (i1) -> i32
  // CHECK-NEXT: %[[AND1:.*]] = aig.and_inv %[[REPLICATE0]], %high : i32
  // CHECK-NEXT: %[[AND2:.*]] = aig.and_inv not %[[REPLICATE0]], %low : i32
  // CHECK-NEXT: %[[AND3:.*]] = aig.and_inv not %[[AND1]], not %[[AND2]] : i32
  // CHECK-NEXT: %[[AND4:.*]] = aig.and_inv not %[[AND3]] : i32
  // CHECK-NEXT: hw.output %[[AND4]] : i32
  %0 = comb.mux %cond, %high, %low : i32
  hw.output %0 : i32
}


// CHECK-LABEL: @datapass
hw.module @datapass(in %arg0: i32, in %arg1: i1, out out: i2) {
  // CHECK-NEXT: %[[EXTRACT0:.*]] = comb.extract %arg0 from 2 : (i32) -> i2
  // CHECK-NEXT: %[[REPLICATE1:.*]] = comb.replicate %arg1 : (i1) -> i2
  // CHECK-NEXT: %[[CONCAT2:.*]] = comb.concat %arg1, %arg1 : i1, i1
  // CHECK-NEXT: %[[AND3:.*]] = aig.and_inv %[[EXTRACT0]], %[[REPLICATE1]], %[[CONCAT2]] : i2
  // CHECK-NEXT: hw.output %[[AND3]] : i2
  %0 = comb.extract %arg0 from 2 : (i32) -> i2
  %1 = comb.replicate %arg1 : (i1) -> i2
  %2 = comb.concat %arg1, %arg1 : i1, i1
  %3 = comb.and %0, %1, %2 : i2
  hw.output %3 : i2
}