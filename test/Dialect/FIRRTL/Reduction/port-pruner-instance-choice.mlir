// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-opt -pass-pipeline='builtin.module(firrtl-imdeadcodeelim{remove-ports-only})' %s -o /dev/null
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "firrtl.module private @Divider_FPGA" --keep-best=0 --include firrtl-imdeadcodeelim-remove-ports | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.option @Target attributes {sym_visibility = "private"} {
    firrtl.option_case @FPGA
  }
  // CHECK-LABEL: firrtl.extmodule private @Divider_ASIC
  // CHECK-SAME:    in in:
  // CHECK-SAME:    out out:
  firrtl.extmodule private @Divider_ASIC(
    in in: !firrtl.clock,
    out out: !firrtl.clock
  )
  // CHECK-LABEL: firrtl.module private @Divider_FPGA
  // CHECK-SAME:    in %in:
  // CHECK-SAME:    out %out:
  firrtl.module private @Divider_FPGA(
    in %in: !firrtl.clock,
    out %out: !firrtl.clock
  ) {
  }
  // CHECK-LABEL: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK: %divider_in, %divider_out = firrtl.instance_choice divider @Divider_ASIC
    // CHECK-SAME: @FPGA -> @Divider_FPGA
    // CHECK-SAME: in in:
    // CHECK-SAME: out out:
    %divider_in, %divider_out = firrtl.instance_choice divider @Divider_ASIC alternatives @Target { @FPGA -> @Divider_FPGA} (in in: !firrtl.clock, out out: !firrtl.clock)
    %invalid_clock = firrtl.invalidvalue : !firrtl.clock
    firrtl.matchingconnect %divider_in, %invalid_clock : !firrtl.clock
  }
}
