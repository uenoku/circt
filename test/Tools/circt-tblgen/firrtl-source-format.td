// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.first
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.second
// RUN: diff %t.first %t.second

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// The directive expands all operands, but not the integer attributes.
def DirectiveOp : FIRRTLOp<"directive"> {
  let arguments = (ins IntType:$lhs, IntType:$rhs, I32Attr:$first, I32Attr:$second);
  let results = (outs IntType:$result);
  let firrtlFormat = "`directive` `(` operands `,` $first `,` $second `)`";
}

// Explicit argument references and shared import/export metadata.
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

// The call shorthand derives its spelling from the mnemonic and still uses
// the shared version and feature metadata.
def ShortVersionedOp : FIRRTLOp<"short_versioned"> {
  let arguments = (ins IntType:$lhs, IntType:$rhs);
  let results = (outs IntType:$result);
  let firrtlFormat = "call";
  let firrtlVersion = FIRRTLReleasedVersion<4, 0, 0>;
  let firrtlFeature = "short feature";
}

def EmptyFormatOp : FIRRTLOp<"empty.format"> {
  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
}

// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(directive, DirectiveOp, 2, 2, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(format_example, FormatExampleOp, 1, 2, FIRVersion(6, 0, 0), "Example feature")
// CHECK: TOK_LPKEYWORD_PRIM(next_format, NextFormatOp, 1, 0, nextFIRVersion, "next feature")
// CHECK: TOK_LPKEYWORD_PRIM(short_versioned, ShortVersionedOp, 2, 0, FIRVersion(4, 0, 0), "short feature")
// CHECK-NOT: EmptyFormatOp
// CHECK: void emitExpression(DirectiveOp op) {
// CHECK-NEXT: emitPrimExpr("directive", op, {op.getFirst(), op.getSecond()});
// CHECK: void emitExpression(FormatExampleOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "Example feature"))) return;
// CHECK-NEXT: emitPrimExpr("format_example", op, {op.getUpper(), op.getLower()});
// CHECK: void emitExpression(NextFormatOp op) {
// CHECK-NEXT: if (failed(requireVersion(nextFIRVersion, op, "next feature"))) return;
// CHECK-NEXT: emitPrimExpr("next_format", op);
// CHECK: void emitExpression(ShortVersionedOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "short feature"))) return;
// CHECK-NEXT: emitPrimExpr("short_versioned", op);
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: DirectiveOp,
// CHECK: FormatExampleOp,
// CHECK: ShortVersionedOp,
// CHECK-NOT: EmptyFormatOp
