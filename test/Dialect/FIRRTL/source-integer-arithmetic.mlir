// RUN: circt-translate --export-firrtl --firrtl-version=4.0.0 %s | FileCheck %s --check-prefix=SOURCE
// RUN: circt-translate --export-firrtl --firrtl-version=7.0.0 %s | FileCheck %s --check-prefix=SOURCE
// RUN: circt-translate --export-firrtl --firrtl-version=4.0.0 %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT
// RUN: circt-translate --export-firrtl --firrtl-version=4.0.0 --target-line-length=10 %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

firrtl.circuit "Arithmetic" {
  firrtl.module @Arithmetic(in %a : !firrtl.integer, in %b : !firrtl.integer,
                             out %addOut : !firrtl.integer, out %mulOut : !firrtl.integer,
                             out %shlOut : !firrtl.integer, out %shrOut : !firrtl.integer) {
    %add = firrtl.integer.add %a, %b : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    %mul = firrtl.integer.mul %add, %b : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    %shl = firrtl.integer.shl %b, %a : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    %shr = firrtl.integer.shr %a, %b : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
    firrtl.propassign %addOut, %add : !firrtl.integer
    firrtl.propassign %mulOut, %mul : !firrtl.integer
    firrtl.propassign %shlOut, %shl : !firrtl.integer
    firrtl.propassign %shrOut, %shr : !firrtl.integer
  }
}

// SOURCE: propassign addOut, integer_add(a, b)
// SOURCE: propassign mulOut, integer_mul(integer_add(a, b), b)
// SOURCE: propassign shlOut, integer_shl(b, a)
// SOURCE: propassign shrOut, integer_shr(a, b)

// IMPORT: firrtl.integer.add %a, %b
// IMPORT: firrtl.propassign %addOut,
// IMPORT: [[ADD:%.*]] = firrtl.integer.add %a, %b
// IMPORT: firrtl.integer.mul [[ADD]], %b
// IMPORT: firrtl.propassign %mulOut,
// IMPORT: firrtl.integer.shl %b, %a
// IMPORT: firrtl.propassign %shlOut,
// IMPORT: firrtl.integer.shr %a, %b
// IMPORT: firrtl.propassign %shrOut,
