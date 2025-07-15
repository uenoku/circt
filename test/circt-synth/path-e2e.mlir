// RUN: circt-synth %s -output-longest-path=- -top counter | FileCheck %s
// RUN: circt-synth %s -output-longest-path=- -top counter -output-longest-path-json | FileCheck %s --check-prefix JSON

// CHECK-LABEL: # Longest Path Analysis result for "counter"
// CHECK-NEXT: Found 189 paths
// CHECK-NEXT: Found 32 unique fanout points
// CHECK-NEXT: Maximum path delay: 48
// Don't test detailed reports as they are not stable.

// Make sure json is emitted.
// JSON: {"module_name":"counter","timing_levels":[
hw.module @counter(in %a: i2, in %b: i2, out result: i2) {
    %add = comb.add %b, %a : i2
    hw.output %add : i2
}
