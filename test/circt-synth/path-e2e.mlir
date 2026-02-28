// RUN: circt-synth %s -analysis-output=- -top counter --enable-sop-balancing | FileCheck %s --check-prefixes COMMON,AIG
// RUN: circt-synth %s -analysis-output=- -top counter --target-ir mig | FileCheck %s --check-prefixes COMMON,MIG
// RUN: circt-synth %s -analysis-output=- -top counter -lower-to-k-lut 6 | FileCheck %s --check-prefixes COMMON,LUT6
// RUN: circt-synth %s -analysis-output=- -top test -analysis-output-format=json | FileCheck %s --check-prefix JSON

// COMMON-LABEL: # Longest Path Analysis result for "counter"
// COMMON-NEXT: Found 168 paths
// COMMON-NEXT: Found 32 unique end points
// AIG-NEXT: Maximum path delay: 27
// MIG-NEXT: Maximum path delay: 32
// LUT6-NEXT: Maximum path delay: 6
// Don't test detailed reports as they are not stable.

hw.module @counter(in %a: i128, in %b: i128, in %c: i128, out result: i128) {
    %add = comb.mul %a, %b, %c : i128
    %mul = comb.add %add, %c, %b : i128
    hw.output %mul : i128
}

