// RUN: circt-opt --synth-sort-topologically %s | FileCheck %s

hw.module.extern @inv(in %a: i1, out y: i1)

// CHECK-LABEL: hw.module @shuffled
hw.module @shuffled(in %a: i1, out y: i1) {
  // User before driver: sorted so the driver comes first.
  // CHECK: hw.instance "driver" @inv
  // CHECK: hw.instance "user" @inv
  %u = hw.instance "user" @inv(a: %d: i1) -> (y: i1)
  %d = hw.instance "driver" @inv(a: %a: i1) -> (y: i1)
  hw.output %u : i1
}
