// RUN: circt-translate --export-firrtl %s | FileCheck %s --check-prefix=SOURCE
// RUN: circt-translate --export-firrtl %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT
// RUN: circt-translate --export-firrtl --target-line-length=10 %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

firrtl.circuit "Test" {
  firrtl.module @Test(in %x : !firrtl.uint<8>, out %out : !firrtl.uint<3>) {
    %0 = firrtl.bits %x 5 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>
    firrtl.connect %out, %0 : !firrtl.uint<3>
  }
}

// SOURCE: connect out, bits(x, 5, 3)
// IMPORT: firrtl.bits %{{.*}} 5 to 3 : (!firrtl.uint<8>) -> !firrtl.uint<3>
