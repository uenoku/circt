// RUN: circt-opt --lower-firrtl-to-hw -split-input-file %s | FileCheck %s
// RUN: circt-opt --lower-firrtl-to-hw="disallow-instance-choice-default" -split-input-file %s | FileCheck %s --check-prefix=NODEFAULT

// Test basic instance choice lowering with single option
// CHECK-LABEL: hw.module @SingleOption
firrtl.circuit "SingleOption" {
  sv.macro.decl @targets$Platform$FPGA
  firrtl.option @Platform {
    firrtl.option_case @FPGA { case_macro = @targets$Platform$FPGA }
  }

  sv.macro.decl @targets$Platform$SingleOption$inst

  firrtl.module private @DefaultMod() {}
  firrtl.module private @FPGAMod() {}

  firrtl.module @SingleOption() {
    // CHECK:      sv.ifdef @targets$Platform$FPGA {
    // CHECK-NEXT:   hw.instance "inst_FPGA" sym @{{.+}} @FPGAMod
    // CHECK-NEXT:   sv.macro.def @targets$Platform$SingleOption$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@SingleOption::@{{.+}}>])
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   hw.instance "inst_default" sym @{{.+}} @DefaultMod
    // CHECK-NEXT:   sv.ifdef @targets$Platform$SingleOption$inst {
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     sv.macro.def @targets$Platform$SingleOption$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@SingleOption::@{{.+}}>])
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.instance_choice inst {instance_macro = @targets$Platform$SingleOption$inst} @DefaultMod alternatives @Platform { @FPGA -> @FPGAMod } ()
  }
}

// -----

// Test instance choice with multiple options
// CHECK-LABEL: hw.module @MultipleOptions
firrtl.circuit "MultipleOptions" {
  sv.macro.decl @targets$Platform$ASIC
  sv.macro.decl @targets$Platform$FPGA
  firrtl.option @Platform {
    firrtl.option_case @ASIC { case_macro = @targets$Platform$ASIC }
    firrtl.option_case @FPGA { case_macro = @targets$Platform$FPGA }
  }

  sv.macro.decl @targets$Platform$MultipleOptions$inst

  firrtl.module private @DefaultMod() {}
  firrtl.module private @ASICMod() {}
  firrtl.module private @FPGAMod() {}

  firrtl.module @MultipleOptions() {
    // CHECK:      sv.ifdef @targets$Platform$ASIC {
    // CHECK-NEXT:   hw.instance "inst_ASIC" sym @{{.+}} @ASICMod
    // CHECK-NEXT:   sv.macro.def @targets$Platform$MultipleOptions$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@MultipleOptions::@{{.+}}>])
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   sv.ifdef @targets$Platform$FPGA {
    // CHECK-NEXT:     hw.instance "inst_FPGA" sym @{{.+}} @FPGAMod
    // CHECK-NEXT:     sv.macro.def @targets$Platform$MultipleOptions$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@MultipleOptions::@{{.+}}>])
    // CHECK-NEXT:   } else {
    // CHECK-NEXT:     hw.instance "inst_default" sym @{{.+}} @DefaultMod
    // CHECK-NEXT:     sv.ifdef @targets$Platform$MultipleOptions$inst {
    // CHECK-NEXT:     } else {
    // CHECK-NEXT:       sv.macro.def @targets$Platform$MultipleOptions$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@MultipleOptions::@{{.+}}>])
    // CHECK-NEXT:     }
    // CHECK-NEXT:   }
    // CHECK-NEXT: }
    firrtl.instance_choice inst {instance_macro = @targets$Platform$MultipleOptions$inst} @DefaultMod alternatives @Platform { @ASIC -> @ASICMod, @FPGA -> @FPGAMod } ()
  }
}

// -----

// Test instance choice with disallow-default option (single option)
// NODEFAULT-LABEL: hw.module @NoDefaultSingle
firrtl.circuit "NoDefaultSingle" {
  sv.macro.decl @targets$Platform$FPGA
  firrtl.option @Platform {
    firrtl.option_case @FPGA { case_macro = @targets$Platform$FPGA }
  }

  sv.macro.decl @targets$Platform$NoDefaultSingle$inst

  firrtl.module private @DefaultMod() {}
  firrtl.module private @FPGAMod() {}

  firrtl.module @NoDefaultSingle() {
    // NODEFAULT:      sv.ifdef @targets$Platform$FPGA {
    // NODEFAULT-NEXT:   hw.instance "inst_FPGA" sym @{{.+}} @FPGAMod
    // NODEFAULT-NEXT:   sv.macro.def @targets$Platform$NoDefaultSingle$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@NoDefaultSingle::@{{.+}}>])
    // NODEFAULT-NEXT: } else {
    // NODEFAULT-NEXT:   sv.error "No valid instance choice case for option Platform, set a macro to one of [targets$Platform$FPGA]"
    // NODEFAULT-NEXT: }
    firrtl.instance_choice inst {instance_macro = @targets$Platform$NoDefaultSingle$inst} @DefaultMod alternatives @Platform { @FPGA -> @FPGAMod } ()
  }
}

// -----

// Test instance choice with disallow-default option (multiple options)
// NODEFAULT-LABEL: hw.module @NoDefaultMultiple
firrtl.circuit "NoDefaultMultiple" {
  sv.macro.decl @targets$Platform$ASIC
  sv.macro.decl @targets$Platform$FPGA
  firrtl.option @Platform {
    firrtl.option_case @ASIC { case_macro = @targets$Platform$ASIC }
    firrtl.option_case @FPGA { case_macro = @targets$Platform$FPGA }
  }

  sv.macro.decl @targets$Platform$NoDefaultMultiple$inst

  firrtl.module private @DefaultMod() {}
  firrtl.module private @ASICMod() {}
  firrtl.module private @FPGAMod() {}

  firrtl.module @NoDefaultMultiple() {
    // NODEFAULT:      sv.ifdef @targets$Platform$ASIC {
    // NODEFAULT-NEXT:   hw.instance "inst_ASIC" sym @{{.+}} @ASICMod
    // NODEFAULT-NEXT:   sv.macro.def @targets$Platform$NoDefaultMultiple$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@NoDefaultMultiple::@{{.+}}>])
    // NODEFAULT-NEXT: } else {
    // NODEFAULT-NEXT:   sv.ifdef @targets$Platform$FPGA {
    // NODEFAULT-NEXT:     hw.instance "inst_FPGA" sym @{{.+}} @FPGAMod
    // NODEFAULT-NEXT:     sv.macro.def @targets$Platform$NoDefaultMultiple$inst "{{[{][{]}}0{{[}][}]}}"([#hw.innerNameRef<@NoDefaultMultiple::@{{.+}}>])
    // NODEFAULT-NEXT:   } else {
    // NODEFAULT-NEXT:     sv.error "No valid instance choice case for option Platform, set a macro to one of [targets$Platform$ASIC, targets$Platform$FPGA]"
    // NODEFAULT-NEXT:   }
    // NODEFAULT-NEXT: }
    firrtl.instance_choice inst {instance_macro = @targets$Platform$NoDefaultMultiple$inst} @DefaultMod alternatives @Platform { @ASIC -> @ASICMod, @FPGA -> @FPGAMod } ()
  }
}

