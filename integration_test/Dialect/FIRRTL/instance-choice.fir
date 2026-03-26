; This test verifies that the instance choice macro-based mechanism works correctly:
; REQUIRES: verilator

; -------------------------------------- Setup
; RUN: rm -rf %t && mkdir -p %t
; RUN: firtool %s --split-verilog -o=%t

; -------------------------------------- Test 1: ASIC target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.asic.exe --top-module top -Dtargets\$Platform\$ASIC
; RUN: %t.asic.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=ASIC
; ASIC: result:         10

; -------------------------------------- Test 2: FPGA target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.fpga.exe --top-module top -Dtargets\$Platform\$FPGA
; RUN: %t.fpga.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=FPGA
; FPGA: result:         20

; -------------------------------------- Test 3: Default target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.default.exe --top-module top
; RUN: %t.default.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=DEFAULT
; DEFAULT: result:       0


FIRRTL version 5.1.0
circuit top:
  option Platform:
    FPGA
    ASIC

  module DefaultTarget:
    output result: UInt<32>

    connect result, UInt<32>(0)

  module ASICTarget:
    output result: UInt<32>

    connect result, UInt<32>(10)

  module FPGATarget:
    output result: UInt<32>

    connect result, UInt<32>(20)

  public module top:
    input clk: Clock
    input rst: UInt<1>

    instchoice proc of DefaultTarget, Platform:
      ASIC => ASICTarget
      FPGA => FPGATarget

    printf(clk, UInt<1>(1), "result: %d\n", proc.result)
