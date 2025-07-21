// RUN: circt-opt --pass-pipeline='builtin.module(synthesis-tech-mapper)' %s --split-input-file --verify-diagnostics

hw.module @do_nothing(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1]]}} {
    hw.output %a : i1
}

hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error-re@+1 {{No matching cut found for value: {{.*}}}}
    %0 = aig.and_inv %a, %b : i1
    hw.output %0 : i1
}

// -----

hw.module @do_nothing(in %a : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1]]}} {
    hw.output %a : i1
}

// No tech library found so pattern matching is failed.
hw.module @test(in %a : i1, in %b : i1, out result : i1) {
    // expected-error-re@+1 {{Cut enumeration supports at most 2 operands, found: 3}}
    %0 = aig.and_inv %a, %b, %a : i1
    hw.output %0 : i1
}


// -----

// expected-error@+1 {{Cut rewriter does not support patterns with multiple outputs yet}}
hw.module @multi_output(in %a : i1, out result1 : i1, out result2 : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
    hw.output %a, %a : i1, i1
}

hw.module @test(in %a : i1, out result : i1) {
    hw.output %a : i1
}


// -----

// expected-error@+1 {{All input ports must be single bit}}
hw.module @multibit(in %a : i2, in %b: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  hw.output %b: i1
}

// -----


// expected-error@+1 {{All input ports must be single bit}}
hw.module @multibit(in %a : i1, in %b: i2, out result : i2) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  hw.output %b: i2
}

// -----


hw.module @multibit(in %a : i1, in %b: i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [2]]}} {
  // expected-error@+1 {{Failed to evaluate a truth table}}
  %0 = comb.xor %a, %b : i1
  hw.output %0: i1
}
