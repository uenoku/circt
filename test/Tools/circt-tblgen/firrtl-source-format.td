// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.first
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.second
// RUN: diff %t.first %t.second

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// An explicit source spelling and shared import/export metadata.
def FormatExampleOp : FIRRTLOp<"different.mnemonic"> {
  let arguments = (ins IntType:$input, I32Attr:$upper, I32Attr:$lower);
  let results = (outs IntType:$result);
  let firrtlFormat = "`format_example` `(` $input `,` $upper `,` $lower `)`";
  let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
  let firrtlFeature = "Example feature";
}

def NextFormatOp : FIRRTLOp<"next.format"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let firrtlFormat = "`next_format` `(` $input `)`";
  let firrtlVersion = FIRRTLNextVersion;
  let firrtlFeature = "next feature";
}

def EmptyFormatOp : FIRRTLOp<"empty.format"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}

// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(format_example, FormatExampleOp, 1, 2, FIRVersion(6, 0, 0), "Example feature")
// CHECK: TOK_LPKEYWORD_PRIM(next_format, NextFormatOp, 1, 0, nextFIRVersion, "next feature")
// CHECK-NOT: EmptyFormatOp
// CHECK: void emitExpression(FormatExampleOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "Example feature"))) return;
// CHECK-NEXT: emitPrimExpr("format_example", op, {op.getUpper(), op.getLower()});
// CHECK: void emitExpression(NextFormatOp op) {
// CHECK-NEXT: if (failed(requireVersion(nextFIRVersion, op, "next feature"))) return;
// CHECK-NEXT: emitPrimExpr("next_format", op);
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: FormatExampleOp,
// CHECK-NOT: EmptyFormatOp
