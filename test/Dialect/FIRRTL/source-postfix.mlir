// RUN: circt-translate --export-firrtl %s | FileCheck %s --check-prefix=SOURCE
// RUN: circt-translate --export-firrtl %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT
// RUN: circt-translate --export-firrtl --target-line-length=10 %s | circt-translate --import-firrtl | FileCheck %s --check-prefix=IMPORT

firrtl.circuit "Postfix" {
  firrtl.module @Postfix(in %agg : !firrtl.bundle<"0": vector<bundle<leaf: uint<8>>, 2>>,
                          in %idx : !firrtl.uint<1>, out %a : !firrtl.uint<8>,
                          out %b : !firrtl.uint<8>) {
    %vector = firrtl.subfield %agg["0"] : !firrtl.bundle<"0": vector<bundle<leaf: uint<8>>, 2>>
    %static = firrtl.subindex %vector[1] : !firrtl.vector<bundle<leaf: uint<8>>, 2>
    %dynamic = firrtl.subaccess %vector[%idx] : !firrtl.vector<bundle<leaf: uint<8>>, 2>, !firrtl.uint<1>
    %staticLeaf = firrtl.subfield %static[leaf] : !firrtl.bundle<leaf: uint<8>>
    %dynamicLeaf = firrtl.subfield %dynamic[leaf] : !firrtl.bundle<leaf: uint<8>>
    firrtl.connect %a, %staticLeaf : !firrtl.uint<8>
    firrtl.connect %b, %dynamicLeaf : !firrtl.uint<8>
  }
}

// SOURCE: connect a, agg.`0`[1].leaf
// SOURCE: connect b, agg.`0`[idx].leaf

// IMPORT: [[VECTOR:%.*]] = firrtl.subfield %agg["0"]
// IMPORT: [[STATIC:%.*]] = firrtl.subindex [[VECTOR]][1]
// IMPORT: [[STATIC_LEAF:%.*]] = firrtl.subfield [[STATIC]][leaf]
// IMPORT: firrtl.matchingconnect %a, [[STATIC_LEAF]]
// IMPORT: [[DYNAMIC:%.*]] = firrtl.subaccess [[VECTOR]][%idx]
// IMPORT: [[DYNAMIC_LEAF:%.*]] = firrtl.subfield [[DYNAMIC]][leaf]
// IMPORT: firrtl.matchingconnect %b, [[DYNAMIC_LEAF]]
