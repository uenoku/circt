; RUN: split-file %s %t
; RUN: firtool %t/parent.fir -output-final-mlir=%t/parent.mlir -domain-mode=infer -strip-domain=false -o %t/parent.sv
; RUN: firtool %t/child.fir -output-final-mlir=%t/child.mlir -domain-mode=infer -strip-domain=false -o %t/child.sv
; RUN: not om-linker %t/parent.mlir %t/child.mlir 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not om-linker %t/child.mlir %t/parent.mlir 2>&1 | FileCheck %s --check-prefix=ERROR

; ERROR: OM property assertion failed: reset must be cold

;--- child.fir
FIRRTL version 7.0.0
circuit Child:
  domain ResetDomain:
    isColdReset: Bool
  public module Child:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    propassert d.isColdReset, String("reset must be cold")

;--- parent.fir
FIRRTL version 7.0.0
circuit Parent:
  domain ResetDomain:
    isColdReset: Bool
  extmodule Child:
    input d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
  public module Parent:
    output d: Domain of ResetDomain
    input rst: AsyncReset domains [d]
    domain resetDomain of ResetDomain(Bool(false))
    domain_define d = resetDomain
    inst child of Child
    connect child.rst, rst
