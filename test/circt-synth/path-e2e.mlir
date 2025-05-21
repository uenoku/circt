// RUN: circt-synth %s -output-longest-paths=%t -top counter && cat %t | FileCheck %s
// CHECK-LABEL: # Analysis result for "counter"
// CHECK-NEXT: Found 135 closed paths
// CHECK-LABEL: ## Percentile Summary
// CHECK-LABEL: | Percentile | Delay | Path |
// CHECK-NEXT:  |------------|-------|------|
// CHECK-NEXT: | 100.0% | 75 | {{.*}}
// CHECK-NEXT: | 99.9% | 75 | {{.*}}
// CHECK-NEXT: | 99.0% | 75 | {{.*}}
// CHECK-NEXT: | 95.0% | 64 | {{.*}}
// CHECK-NEXT: | 90.0% | 56 | {{.*}}
// CHECK-NEXT: | 50.0% | 30 | {{.*}}
// CHECK-LABEL: ## Top 10 (5% of 135 paths) Closed Paths
// CHECK-LABEL: | Rank | Delay | Path |
// CHECK-NEXT:  |------|-------|------|
// CHECK-NEXT:  | 1 | 75 | {{.*}}
// CHECK:       | 10 | 62 | {{.*}}
hw.module @counter(in %a: i16, in %clk: !seq.clock, out result: i16) {
    %reg = seq.compreg %add, %clk : i16
    %add = comb.mul %reg, %a : i16
    hw.output %reg : i16
}
