; RUN: split-file %s %t
; RUN: firtool %t/tb.fir -output-final-mlir=%t/tb.mlir -domain-mode=infer -strip-domain=false -o %t/tb.sv
; RUN: firtool %t/dut.fir -output-final-mlir=%t/dut.mlir -domain-mode=infer -strip-domain=false -o %t/dut.sv
; RUN: not om-linker %t/tb.mlir %t/dut.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not om-linker %t/dut.mlir %t/tb.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: om-linker %t/tb.mlir %t/dut.mlir --disable-elaboration | FileCheck %s

; ERROR: OM property assertion failed: reset must be cold

; CHECK-DAG: om.class @testbench_Class(%{{.*}}: !om.frozenbasepath) -> (d_out: !om.class.type<@ResetDomain_out>)
; CHECK-DAG: om.class @Dut_Class(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ResetDomain>) -> (d_out: !om.class.type<@ResetDomain_out>)
; CHECK-DAG: om.class @ResetDomain(%{{.*}}: !om.frozenbasepath, %{{.*}}: i1) -> (isColdReset_out: i1)
; CHECK-DAG: om.class @ResetDomain_out(%{{.*}}: !om.frozenbasepath, %{{.*}}: !om.class.type<@ResetDomain>, %{{.*}}: !om.list<!om.frozenpath>) -> (domainInfo_out: !om.class.type<@ResetDomain>, associations_out: !om.list<!om.frozenpath>)
; CHECK-DAG: %[[COLD:.+]] = om.object.field %{{.+}}["isColdReset_out"] : (!om.class.type<@ResetDomain>) -> i1
; CHECK-DAG: om.property_assert %[[COLD]], %{{.+}} : i1

;--- dut.fir
FIRRTL version 100.0.0
circuit Dut:
  domain ResetDomain:
    isColdReset: Bool
  public module Dut:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    output out: UInt<1> domains [d]
    propassert d.isColdReset, String("reset must be cold")
    connect out, UInt<1>(0)

;--- tb.fir
FIRRTL version 100.0.0
circuit testbench:
  domain ResetDomain:
    isColdReset: Bool
  extmodule Dut:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    output out: UInt<1> domains [d]
  public module testbench:
    output d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    domain resetDomain of ResetDomain(Bool(false))
    domain_define d = resetDomain
    inst dut of Dut
    connect dut.rst, rst
