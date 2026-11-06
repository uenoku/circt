; RUN: circt-translate --import-firrtl %s | FileCheck %s

FIRRTL version 2.0.0
circuit Test :
  module Test :
    input enable : UInt<1>
    output skip : UInt<1>
    output attach : UInt<1>
    skip <= enable
    attach <= enable
    skip

; CHECK: firrtl.matchingconnect %skip, %enable
; CHECK: firrtl.matchingconnect %attach, %enable
; CHECK: firrtl.skip
