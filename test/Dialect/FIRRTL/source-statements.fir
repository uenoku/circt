; RUN: circt-translate --import-firrtl %s | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl | FileCheck %s --check-prefix=EXPORT

FIRRTL version 7.0.0
circuit Test :
  public module Test :
    input clock : Clock
    input enable : UInt<1>
    input a : Analog<1>
    input b : Analog<1>
    output skip : UInt<1>
    connect skip, enable
    skip
    attach(a, b)
    printf(clock, enable, "hello")
    fprintf(clock, enable, "out.txt", "hello")
    fflush(clock, enable)
    stop(clock, enable, 1)

; IMPORT: firrtl.matchingconnect %skip, %enable
; IMPORT: firrtl.skip
; IMPORT: firrtl.attach %a, %b
; IMPORT: firrtl.printf
; IMPORT: firrtl.fprintf
; IMPORT: firrtl.fflush
; IMPORT: firrtl.stop
; EXPORT: connect skip, enable
; EXPORT: skip
; EXPORT: attach(a, b)
; EXPORT: printf(clock, enable, "hello")
; EXPORT: fprintf(clock, enable, "out.txt", "hello")
; EXPORT: fflush(clock, enable)
; EXPORT: stop(clock, enable, 1)
