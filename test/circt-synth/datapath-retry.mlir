// RUN: mkdir -p %t/bin && cp %S/Inputs/fake-sta.sh %t/bin/sta && chmod +x %t/bin/sta && env PATH=%t/bin:$PATH circt-synth --liberty %S/Inputs/datapath-retry.lib --strip-lib-to-extern --top mul4 --datapath-retry=2 --retry-timeout=120 %s -o %t/out.mlir 2>&1 | FileCheck %s

// In-driver datapath retry with a hermetic fake STA: baseline violates
// (slack -5), force-booth satisfies (+2), compress-to-add violates (-1).
// Satisfying + min-area wins, so force-booth must be BEST; the winner's
// MLIR lands on -o.
// CHECK: datapath retry results
// CHECK: [    ] baseline
// CHECK: [BEST] force-booth
// CHECK: slack=2{{.*}} arrival=8
hw.module @mul4(in %a : i4, in %b : i4, out p : i8) {
  %0 = comb.mul %a, %b : i4
  %c0 = hw.constant 0 : i4
  %1 = comb.concat %c0, %0 : i4, i4
  hw.output %1 : i8
}
