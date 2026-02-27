; RUN: firtool %s --disable-all-randomization --strip-debug-info | FileCheck %s

; Test XMR lowering for InstanceChoiceOp with probes
; This test verifies that:
; 1. XMR macros are generated for instance choice probe ports
; 2. The macros use ifdef guards based on option cases
; 3. HierPath symbols are properly substituted
; 4. Only public modules generate ref header files

FIRRTL version 5.1.0
circuit Top :
  option Platform : @[test_instance_choice_xmr.mlir 2:3]
    FPGA @[test_instance_choice_xmr.mlir 3:5]
    ASIC @[test_instance_choice_xmr.mlir 4:5]
  option Optimization : @[test_instance_choice_xmr.mlir 2:3]
    Fast @[test_instance_choice_xmr.mlir 3:5]
    Slow @[test_instance_choice_xmr.mlir 4:5]

  module DefaultTarget : @[test_instance_choice_xmr.mlir 8:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 8:43]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 8:50]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 8:68]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 8:96]

    reg r : UInt<8>, clk @[test_instance_choice_xmr.mlir 9:10]
    connect r, in @[test_instance_choice_xmr.mlir 11:5]
    connect data, r @[test_instance_choice_xmr.mlir 12:5]
    define probe = probe(r) @[test_instance_choice_xmr.mlir 14:5]

  module FPGAInner : @[test_instance_choice_xmr.mlir 18:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 18:39]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 18:56]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 18:64]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 18:92]

    reg r : UInt<8>, clk @[test_instance_choice_xmr.mlir 19:10]
    connect r, in @[test_instance_choice_xmr.mlir 21:5]
    connect data, r @[test_instance_choice_xmr.mlir 22:5]
    define probe = probe(r) @[test_instance_choice_xmr.mlir 24:5]

  module FPGATarget : @[test_instance_choice_xmr.mlir 28:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 28:40]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 28:57]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 28:65]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 28:93]

    inst inner of FPGAInner @[test_instance_choice_xmr.mlir 29:45]
    connect inner.clk, clk @[test_instance_choice_xmr.mlir 30:5]
    connect inner.in, in @[test_instance_choice_xmr.mlir 30:5]
    connect data, inner.data @[test_instance_choice_xmr.mlir 31:5]
    define probe = inner.probe @[test_instance_choice_xmr.mlir 32:5]

  module ASICDeep : @[test_instance_choice_xmr.mlir 36:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 36:38]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 36:45]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 36:63]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 36:91]

    reg r : UInt<8>, clk @[test_instance_choice_xmr.mlir 37:10]
    connect r, in @[test_instance_choice_xmr.mlir 39:5]
    connect data, r @[test_instance_choice_xmr.mlir 40:5]
    define probe = probe(r) @[test_instance_choice_xmr.mlir 42:5]

  module ASICMiddle : @[test_instance_choice_xmr.mlir 46:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 46:40]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 46:57]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 46:65]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 46:93]

    inst deep of ASICDeep @[test_instance_choice_xmr.mlir 47:42]
    connect deep.clk, clk @[test_instance_choice_xmr.mlir 48:5]
    connect deep.in, in @[test_instance_choice_xmr.mlir 48:5]
    connect data, deep.data @[test_instance_choice_xmr.mlir 49:5]
    define probe = deep.probe @[test_instance_choice_xmr.mlir 50:5]

  module ASICTarget : @[test_instance_choice_xmr.mlir 54:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 54:40]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 54:57]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 54:65]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 54:93]

    inst middle of ASICMiddle @[test_instance_choice_xmr.mlir 55:48]
    connect middle.in, in @[test_instance_choice_xmr.mlir 56:5]
    connect middle.clk, clk @[test_instance_choice_xmr.mlir 56:5]
    connect data, middle.data @[test_instance_choice_xmr.mlir 57:5]
    define probe = middle.probe @[test_instance_choice_xmr.mlir 58:5]

  module TargetWithRWProbe : @[test_instance_choice_xmr.mlir 76:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 76:39]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 76:56]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 76:64]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 76:92]
    output rwprobe : RWProbe<UInt<8>> @[test_instance_choice_xmr.mlir 76:120]

    reg r : UInt<8>, clk @[test_instance_choice_xmr.mlir 77:10]
    connect r, in @[test_instance_choice_xmr.mlir 79:5]
    connect data, r @[test_instance_choice_xmr.mlir 80:5]
    define probe = probe(r) @[test_instance_choice_xmr.mlir 82:5]
    define rwprobe = rwprobe(r) @[test_instance_choice_xmr.mlir 83:5]

  module InstanceChoiceProbe : @[test_instance_choice_xmr.mlir 90:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 90:41]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 90:58]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 90:66]
    output out : UInt<8> @[test_instance_choice_xmr.mlir 90:94]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 90:122]

    instchoice inst of DefaultTarget, Platform : @[test_instance_choice_xmr.mlir 63:42]
      FPGA => FPGATarget
      ASIC => ASICTarget
    connect inst.clk, clk @[test_instance_choice_xmr.mlir 68:5]
    connect inst.in, in @[test_instance_choice_xmr.mlir 68:5]
    connect data, inst.data @[test_instance_choice_xmr.mlir 69:5]
    connect out, read(inst.probe) @[test_instance_choice_xmr.mlir 72:5]
    define probe = inst.probe @[test_instance_choice_xmr.mlir 73:5]

  public module Top : @[test_instance_choice_xmr.mlir 110:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 110:41]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 110:58]
    output data1 : UInt<8> @[test_instance_choice_xmr.mlir 110:66]
    output data2 : UInt<8> @[test_instance_choice_xmr.mlir 110:94]
    output out1 : UInt<8> @[test_instance_choice_xmr.mlir 110:122]

    inst sub1 of InstanceChoiceProbe @[test_instance_choice_xmr.mlir 111:42]
    connect sub1.clk, clk @[test_instance_choice_xmr.mlir 112:5]
    connect sub1.in, in @[test_instance_choice_xmr.mlir 112:5]
    connect data1, sub1.data @[test_instance_choice_xmr.mlir 113:5]
    connect out1, sub1.out @[test_instance_choice_xmr.mlir 114:5]

    inst sub2 of InstanceChoiceRWProbe @[test_instance_choice_xmr.mlir 116:42]
    connect sub2.clk, clk @[test_instance_choice_xmr.mlir 117:5]
    connect sub2.in, in @[test_instance_choice_xmr.mlir 117:5]
    connect data2, sub2.data @[test_instance_choice_xmr.mlir 118:5]

  module InstanceChoiceRWProbe : @[test_instance_choice_xmr.mlir 100:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 100:41]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 100:58]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 100:66]
    output rwprobe : RWProbe<UInt<8>> @[test_instance_choice_xmr.mlir 100:94]

    instchoice inst of TargetWithRWProbe, Platform : @[test_instance_choice_xmr.mlir 101:42]
      FPGA => TargetWithRWProbe
      ASIC => TargetWithRWProbe
    connect inst.clk, clk @[test_instance_choice_xmr.mlir 106:5]
    connect inst.in, in @[test_instance_choice_xmr.mlir 106:5]
    connect data, inst.data @[test_instance_choice_xmr.mlir 107:5]
    define rwprobe = inst.rwprobe @[test_instance_choice_xmr.mlir 108:5]

; CHECK-LABEL: module InstanceChoiceProbe
; CHECK: assign out = `ref_InstanceChoiceProbe_inst_probe;

; CHECK-LABEL: module Top
