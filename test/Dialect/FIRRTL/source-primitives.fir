; RUN: circt-translate --import-firrtl %s | FileCheck %s --check-prefix=IMPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl | FileCheck %s --check-prefix=EXPORT
; RUN: circt-translate --import-firrtl %s | circt-translate --export-firrtl --target-line-length=10 | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

FIRRTL version 4.0.0
circuit Test :
  public module Test :
    input a : UInt<8>
    input b : UInt<8>
    node sum = add(a, b)
    node slice = bits(a, 6, 2)
    node left = shl(a, 3)
    node right = shr(b, 2)
    node nested = bits(add(a, b), 7, 3)

; IMPORT: firrtl.add %a, %b
; IMPORT: firrtl.bits %a 6 to 2
; IMPORT: firrtl.shl %a, 3
; IMPORT: firrtl.shr %b, 2
; IMPORT: firrtl.add %a, %b
; IMPORT: firrtl.bits {{.*}} 7 to 3
; EXPORT: node sum = add(a, b)
; EXPORT: node slice = bits(a, 6, 2)
; EXPORT: node left = shl(a, 3)
; EXPORT: node right = shr(b, 2)
; EXPORT: node nested = bits(add(a, b), 7, 3)
