// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_only,' --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_unary,' --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_single,' --implicit-check-not='TOK_LPKEYWORD_PRIM(read,' --implicit-check-not='TOK_LPKEYWORD_PRIM(probe,' --implicit-check-not='TOK_LPKEYWORD_PRIM(string_concat,'

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

def CustomParserCallOp : FIRRTLOp<"other.call"> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_only` `(` operands `)`";
  let firrtlHasCustomParser = 1;
  let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
}

def CustomParserSingleGroupOp : FIRRTLOp<"other.single"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_single` `(` operands `)`";
  let firrtlHasCustomParser = 1;
}

def CustomParserUnaryOp : FIRRTLOp<"other.unary"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_unary` `(` $value `)`";
  let firrtlHasCustomParser = 1;
}

def TransparentCastOp : FIRRTLOp<"other.cast"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  // A bare operand has no source syntax for reconstructing the cast.
  let firrtlFormat = "$value";
  let firrtlVersion = FIRRTLReleasedVersion<4, 0, 0>;
}

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK-NOT: TransparentCastOp
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(CustomParserCallOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "custom_only"))) return;
// CHECK-NEXT: emitPrimExpr("custom_only", op);
// CHECK: void emitExpression(CustomParserSingleGroupOp op) {
// CHECK-NEXT: ps << "custom_single(";
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK-NEXT: ps << ")";
// CHECK: void emitExpression(CustomParserUnaryOp op) {
// CHECK-NEXT: ps << "custom_unary(";
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK-NEXT: ps << ")";
// CHECK: void emitExpression(TransparentCastOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "other.cast"))) return;
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: CustomParserCallOp,
// CHECK: CustomParserSingleGroupOp,
// CHECK: CustomParserUnaryOp,
// CHECK: TransparentCastOp,
