; RUN: circt-translate --import-firrtl %s | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl --mlir-print-debuginfo %s | circt-translate --export-firrtl --firrtl-version=2.0.0 | FileCheck %s --check-prefix=SOURCE
; RUN: circt-translate --import-firrtl --mlir-print-debuginfo %s | circt-translate --export-firrtl --firrtl-version=2.0.0 | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl --firrtl-version=2.0.0 --target-line-length=10 | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

FIRRTL version 2.0.0
circuit Statements :
  module Statements :
    input clock : Clock
    input enable : UInt<1>
    wire skip : UInt<1>
    skip is invalid
    skip @[Example.scala 3:2]
    stop(clock, enable, 42) : myStop @[Example.scala 4:2]
    stop(clock, enable, 1)

; IMPORT: %skip = firrtl.wire
; IMPORT: firrtl.invalidvalue
; IMPORT: firrtl.skip
; IMPORT: firrtl.stop %clock, %enable, 42 {name = "myStop"}
; IMPORT: firrtl.stop %clock, %enable, 1

; SOURCE: wire skip : UInt<1>
; SOURCE: skip is invalid
; SOURCE: skip @[Example.scala 3:2]
; SOURCE: stop(clock, enable, 42) : myStop @[Example.scala 4:2]
; SOURCE: stop(clock, enable, 1)
