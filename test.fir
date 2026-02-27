FIRRTL version 5.1.0
circuit InstanceChoiceProbe :
  option Platform : @[test_instance_choice_xmr.mlir 2:3]
    FPGA @[test_instance_choice_xmr.mlir 3:5]
    ASIC @[test_instance_choice_xmr.mlir 4:5]

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

  public module InstanceChoiceProbe : @[test_instance_choice_xmr.mlir 62:3]
    input clk : Clock @[test_instance_choice_xmr.mlir 62:41]
    input in: UInt<8> @[test_instance_choice_xmr.mlir 62:58]
    output data : UInt<8> @[test_instance_choice_xmr.mlir 62:66]
    output out : UInt<8> @[test_instance_choice_xmr.mlir 62:94]
    output probe : Probe<UInt<8>> @[test_instance_choice_xmr.mlir 54:93]

    instchoice inst of DefaultTarget, Platform : @[test_instance_choice_xmr.mlir 63:42]
      FPGA => FPGATarget
      ASIC => ASICTarget
    connect inst.clk, clk @[test_instance_choice_xmr.mlir 68:5]
    connect inst.in, in @[test_instance_choice_xmr.mlir 68:5]
    connect data, inst.data @[test_instance_choice_xmr.mlir 69:5]
    connect out, read(inst.probe) @[test_instance_choice_xmr.mlir 72:5]
    define probe = inst.probe @[test_instance_choice_xmr.mlir 72:5]
