// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper{library-modules=and_inv,and_inv_n,and_inv_nn,and_inv_3 strategy=area})' %s | FileCheck %s --check-prefixes CHECK,AREA
// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper{library-modules=and_inv,and_inv_n,and_inv_nn,and_inv_3 strategy=timing})' %s | FileCheck %s --check-prefixes CHECK,TIMING
// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper{library-modules=and_inv,and_inv_n,and_inv_nn,and_inv_3,permutation})' %s | FileCheck %s --check-prefix PERMUTATION

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

// PERMUTATION-LABEL: hw.module @permutation_test(in %p : i1, in %q : i1, in %r : i1, in %s : i1, out result : i1) {
hw.module @permutation_test(in %p: i1, in %q: i1, in %r: i1, in %s: i1, out result: i1) {
    // a -> s
    // b -> p
    // c -> q
    // d -> r
    // PERMUTATION: hw.instance "{{.+}}" @permutation(a: %s: i1, b: %p: i1, c: %q: i1, d: %r: i1) -> (result: i1) {test.arrival_times = [1]}
    %0 = aig.and_inv %s, not %p : i1
    %1 = aig.and_inv %q, not %r : i1
    %2 = aig.and_inv %0, not %1 : i1
    hw.output %2 : i1
}