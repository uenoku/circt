; RUN: split-file %s %t
; RUN: firtool %t/tb.fir -output-final-mlir=%t/tb.mlir -domain-mode=infer -strip-domain=false -o %t/tb.sv
; RUN: firtool %t/dut.fir -output-final-mlir=%t/dut.mlir -domain-mode=infer -strip-domain=false -o %t/dut.sv
; RUN: om-linker %t/tb.mlir %t/dut.mlir | FileCheck %s
; RUN: om-linker %t/dut.mlir %t/tb.mlir | FileCheck %s
; RUN: om-linker %t/tb.mlir %t/dut.mlir --disable-elaboration | FileCheck %s

; CHECK-DAG: om.class @testbench_Class(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ClockDomain>) -> (d_out: !om.class.type<@ClockDomain_out>)
; CHECK-DAG: om.class @Dut_Class(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ClockDomain>) -> (d_out: !om.class.type<@ClockDomain_out>)
; CHECK-DAG: om.class @ClockDomain(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.string) -> (name_out: !om.string)
; CHECK-DAG: om.class @ClockDomain_out(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ClockDomain>, %{{.*}}: !om.list<!om.frozenpath>) -> (domainInfo_out: !om.class.type<@ClockDomain>, associations_out: !om.list<!om.frozenpath>)

;--- dut.fir
FIRRTL version 100.0.0
circuit Dut:
  domain ClockDomain:
    name: String
  public module Dut:
    input d: Domain of ClockDomain
    input clk: Clock domains [d]
    output out: UInt<1> domains [d]
    connect out, UInt<1>(0)

;--- tb.fir
FIRRTL version 100.0.0
circuit testbench:
  domain ClockDomain:
    name: String
  extmodule Dut:
    input d: Domain of ClockDomain
    input clk: Clock domains [d]
    output out: UInt<1> domains [d]
  public module testbench:
    input d: Domain of ClockDomain
    input clk: Clock domains [d]
    inst dut of Dut
    connect dut.clk, clk
