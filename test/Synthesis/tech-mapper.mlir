// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper{strategy=area})' %s | FileCheck %s --check-prefixes CHECK,AREA
// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper{strategy=timing})' %s | FileCheck %s --check-prefixes CHECK,TIMING

hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv not %a, %b : i1
    hw.output %0 : i1
}

hw.module @and_inv_nn(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    hw.output %0 : i1
}

// Delay is shorter than @and_inv + @and_inv_n_n. Area is (significantly) larger than @and_inv_n + @and_inv_n_n.
// Check that we use @and_inv_3 if strategy = timing, and @and_inv_n + @and_inv_n_n if strategy = area.
hw.module @and_inv_3(in %a : i1, in %b : i1, in %c : i1, out result : i1) attributes {hw.techlib.info = {area = 10.0 : f64, delay = [[1], [1], [1]]}} {
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv not %0, %c : i1
    hw.output %1 : i1
}

// CHECK-LABEL: @test_strategy
hw.module @test_strategy(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
    // AREA-NEXT: %[[area_0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // AREA-NEXT: %[[area_1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_n(a: %[[area_0]]: i1, b: %c: i1) -> (result: i1) {test.arrival_times = [2]}
    // AREA-NEXT: hw.output %[[area_1]] : i1
    // TIMING-NEXT: %[[timing:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_3(a: %a: i1, b: %b: i1, c: %c: i1) -> (result: i1) {test.arrival_times = [1]}
    // TIMING-NEXT: hw.output %[[timing]] : i1
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %c, not %0 : i1
    hw.output %1 : i1
}

hw.module @permutation(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out result: i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1], [1], [1]]}} {
    %0 = aig.and_inv %a, not %b : i1
    %1 = aig.and_inv %c, not %d : i1
    %2 = aig.and_inv %0, not %1 : i1
    hw.output %2 : i1
}

// CHECK-LABEL: hw.module @permutation_test(in %p : i1, in %q : i1, in %r : i1, in %s : i1, out result : i1) {
hw.module @permutation_test(in %p: i1, in %q: i1, in %r: i1, in %s: i1, out result: i1) {
    // {a -> s, b -> p, c -> q, d -> r}
    // CHECK-NEXT: hw.instance "{{.+}}" @permutation(a: %s: i1, b: %p: i1, c: %q: i1, d: %r: i1) -> (result: i1) {test.arrival_times = [1]}
    %0 = aig.and_inv %s, not %p : i1
    %1 = aig.and_inv %q, not %r : i1
    %2 = aig.and_inv %0, not %1 : i1
    hw.output %2 : i1
}

hw.module @and_inv_5(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2], [2], [2], [1]]}} {
    %0 = aig.and_inv not %a, %b, not %c, %d, not %e : i1
    hw.output %0 : i1
}

// Make sure truth value is computed correctly for @and_inv_5.
// CHECK-LABEL: @and_inv_5_test
hw.module @and_inv_5_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e: i1, out o1 : i1, out o2 : i1) {
    %0 = aig.and_inv not %a, %b : i1
    %1 = aig.and_inv not %c, %d : i1
    %2 = aig.and_inv %0, %1 : i1
    %3 = aig.and_inv %2, not %e : i1
    // CHECK-NEXT: %[[result_0:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_5(a: %a: i1, b: %b: i1, c: %c: i1, d: %d: i1, e: %e: i1)
    %4 = aig.and_inv not %a, not %d : i1
    %5 = aig.and_inv not %b, %e : i1
    %6 = aig.and_inv %5, %c : i1
    %7 = aig.and_inv %6, %4 : i1
    // CHECK-NEXT: %[[result_1:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_5(a: %b: i1, b: %e: i1, c: %a: i1, d: %c: i1, e: %d: i1)
    
    hw.output %3, %7 : i1, i1
    // CHECK-NEXT: hw.output %[[result_0]], %[[result_1]] : i1, i1
}

hw.module @area_flow(in %a : i1, in %b : i1, in %c: i1, out result : i1) attributes {hw.techlib.info = {area = 1.5 : f64, delay = [[10], [10], [10], [10], [10]]}} {
    %0 = aig.and_inv not %a, not %b : i1
    %1 = aig.and_inv not %c, %0 : i1
    hw.output %1 : i1
}

// This is a test that needs area-flow to get an optimal result.
// It produces sub-optimal mappings since currently area-flow is not implemented.
// CHECK-LABEL: @area_flow_test
hw.module @area_flow_test(in %a : i1, in %b : i1, in %c: i1, out result : i1) {
    // FIXME: If area-flow is implemented, this should be mapped to @area_flow with area strategy.
    // CHECK:       hw.instance {{.*}} @and_inv_nn(
    // CHECK-NEXT:  hw.instance {{.*}} @and_inv_n(
    %0 = aig.and_inv not %a, not %b : i1
    %1 = aig.and_inv not %c, %0 : i1
    hw.output %1 : i1
}

// Test primary inputs handling
// CHECK-LABEL: @primary_inputs_test
hw.module @primary_inputs_test(in %a : i1, in %b : i1, out result : i1) {
    // Simple direct mapping - should use @and_inv
    // CHECK-NEXT: %[[primary:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK-NEXT: hw.output %[[primary]] : i1
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// Test chain of operations for timing analysis
// CHECK-LABEL: @timing_chain_test
hw.module @timing_chain_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out result : i1) {
    // Test that timing is accumulated correctly through the chain
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %c: i1, b: %d: i1) -> (result: i1) {test.arrival_times = [1]}
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %{{.+}}: i1, b: %{{.+}}: i1) -> (result: i1) {test.arrival_times = [2]}
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %c, %d : i1
    %2 = aig.and_inv %0, %1 : i1
    hw.output %2 : i1
}

// Test negation patterns
// CHECK-LABEL: @negation_patterns_test
hw.module @negation_patterns_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, out o1 : i1, out o2 : i1, out o3 : i1) {
    // Test all negation patterns are correctly matched
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv(a: %a: i1, b: %b: i1)
    %0 = aig.and_inv %a, %b : i1
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_n(a: %c: i1, b: %d: i1)
    %1 = aig.and_inv not %c, %d : i1
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_nn(a: %a: i1, b: %c: i1)
    %2 = aig.and_inv not %a, not %c : i1
    hw.output %0, %1, %2 : i1, i1, i1
}

// Test cut enumeration with multiple paths
// CHECK-LABEL: @multi_path_test
hw.module @multi_path_test(in %a : i1, in %b : i1, in %c : i1, out result : i1) {
    // This creates multiple possible cuts - test that best one is selected
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %b, %c : i1
    %2 = aig.and_inv %0, %1 : i1
    // Should prefer smaller cuts over larger ones for area strategy
    hw.output %2 : i1
}

// Test complex permutation matching
hw.module @complex_perm(in %x : i1, in %y : i1, in %z : i1, in %w : i1, out result : i1) attributes {hw.techlib.info = {area = 2.0 : f64, delay = [[1], [1], [1], [1]]}} {
    // Complex pattern: (x & !y) | (!z & w)
    %0 = aig.and_inv %x, not %y : i1
    %1 = aig.and_inv not %z, %w : i1
    %2 = aig.and_inv not %0, not %1 : i1  // De Morgan's law: !(A & B) = !A | !B
    %3 = aig.and_inv not %2 : i1  // Double negation
    hw.output %3 : i1
}

// CHECK-LABEL: @complex_permutation_test
hw.module @complex_permutation_test(in %p : i1, in %q : i1, in %r : i1, in %s : i1, out result : i1) {
    // Test complex permutation: should match @complex_perm with input reordering
    // Pattern: (s & !p) | (!q & r) - maps to complex_perm with {x->s, y->p, z->q, w->r}
    %0 = aig.and_inv %s, not %p : i1
    %1 = aig.and_inv not %q, %r : i1
    %2 = aig.and_inv not %0, not %1 : i1
    %3 = aig.and_inv not %2 : i1
    // CHECK: hw.instance "{{[a-zA-Z0-9_]+}}" @complex_perm(x: %s: i1, y: %p: i1, z: %q: i1, w: %r: i1)
    hw.output %3 : i1
}

// Test deep logic chains
// CHECK-LABEL: @deep_chain_test
hw.module @deep_chain_test(in %a : i1, in %b : i1, in %c : i1, in %d : i1, in %e : i1, out result : i1) {
    // Test that deep chains are handled efficiently
    %0 = aig.and_inv %a, %b : i1
    %1 = aig.and_inv %0, %c : i1
    %2 = aig.and_inv %1, %d : i1
    %3 = aig.and_inv %2, %e : i1
    // Should create a series of 2-input mappings
    hw.output %3 : i1
}

// Test fan-out handling
// CHECK-LABEL: @fanout_test
hw.module @fanout_test(in %a : i1, in %b : i1, in %c : i1, out o1 : i1, out o2 : i1) {
    // Test that intermediate results with fan-out are handled correctly
    %0 = aig.and_inv %a, %b : i1  // This will have fan-out
    %1 = aig.and_inv %0, %c : i1
    %2 = aig.and_inv %0, not %c : i1
    // %0 is used in both %1 and %2, test that it's handled correctly
    hw.output %1, %2 : i1, i1
}

// Test comb.extract and comb.concat handling
// CHECK-LABEL: @extract_concat_test
hw.module @extract_concat_test(in %data : i4, in %ctrl : i2, out result : i3) {
    // Extract individual bits from multi-bit inputs
    %bit0 = comb.extract %data from 0 : (i4) -> i1
    %bit1 = comb.extract %data from 1 : (i4) -> i1
    %bit2 = comb.extract %data from 2 : (i4) -> i1
    %bit3 = comb.extract %data from 3 : (i4) -> i1
    
    %ctrl0 = comb.extract %ctrl from 0 : (i2) -> i1
    %ctrl1 = comb.extract %ctrl from 1 : (i2) -> i1
    
    // Apply some logic using AIG operations
    %and0 = aig.and_inv %bit0, %bit1 : i1
    %and1 = aig.and_inv %bit2, not %ctrl0 : i1
    %and2 = aig.and_inv %bit3, %ctrl1 : i1
    
    // Further logic operations
    %out0 = aig.and_inv %and0, %ctrl0 : i1
    %out1 = aig.and_inv %and1, not %and0 : i1
    %out2 = aig.and_inv %and2, %ctrl1 : i1
    
    // Concatenate results into multi-bit output
    %result_concat = comb.concat %out2, %out1, %out0 : i1, i1, i1
    
    // CHECK: comb.extract %data from 0
    // CHECK: comb.extract %data from 1
    // CHECK: comb.extract %data from 2
    // CHECK: comb.extract %data from 3
    // CHECK: comb.extract %ctrl from 0
    // CHECK: comb.extract %ctrl from 1
    // CHECK: hw.instance {{.*}} @and_inv(a: %{{.+}}: i1, b: %{{.+}}: i1)
    // CHECK: hw.instance {{.*}} @and_inv_n(a: %{{.+}}: i1, b: %{{.+}}: i1)
    // CHECK: hw.instance {{.*}} @and_inv(a: %{{.+}}: i1, b: %{{.+}}: i1)
    // CHECK: hw.instance {{.*}} @and_inv_n(a: %{{.+}}: i1, b: %{{.+}}: i1)
    // CHECK: hw.instance {{.*}} @and_inv(a: %{{.+}}: i1, b: %{{.+}}: i1)
    // CHECK: comb.concat
    
    hw.output %result_concat : i3
}

// Test mixed bit-width operations with tech mapping
// CHECK-LABEL: @mixed_bitwidth_test
hw.module @mixed_bitwidth_test(in %a : i8, in %b : i4, out result : i6) {
    // Extract various bit ranges
    %a_low = comb.extract %a from 0 : (i8) -> i4
    %a_high = comb.extract %a from 4 : (i8) -> i4
    
    %b0 = comb.extract %b from 0 : (i4) -> i1
    %b1 = comb.extract %b from 1 : (i4) -> i1
    %b2 = comb.extract %b from 2 : (i4) -> i1
    %b3 = comb.extract %b from 3 : (i4) -> i1
    
    %a0 = comb.extract %a_low from 0 : (i4) -> i1
    %a1 = comb.extract %a_low from 1 : (i4) -> i1
    %a2 = comb.extract %a_high from 0 : (i4) -> i1
    %a3 = comb.extract %a_high from 1 : (i4) -> i1
    
    // AIG logic on individual bits
    %logic0 = aig.and_inv %a0, %b0 : i1
    %logic1 = aig.and_inv not %a1, %b1 : i1
    %logic2 = aig.and_inv %a2, not %b2 : i1
    %logic3 = aig.and_inv not %a3, not %b3 : i1
    
    // Combine some results
    %combine0 = aig.and_inv %logic0, %logic1 : i1
    %combine1 = aig.and_inv %logic2, %logic3 : i1
    
    // Create multi-bit result
    %result_bits = comb.concat %combine1, %combine0, %logic1, %logic0, %b1, %b0 : i1, i1, i1, i1, i1, i1
    
    // CHECK: comb.extract
    // CHECK: comb.extract
    // CHECK: hw.instance {{.*}} @and_inv
    // CHECK: hw.instance {{.*}} @and_inv_n
    // CHECK: hw.instance {{.*}} @and_inv_n
    // CHECK: hw.instance {{.*}} @and_inv_nn
    // CHECK: hw.instance {{.*}} @and_inv
    // CHECK: hw.instance {{.*}} @and_inv
    // CHECK: comb.concat
    
    hw.output %result_bits : i6
}
