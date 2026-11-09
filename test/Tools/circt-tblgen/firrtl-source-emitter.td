// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_only,' --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_unary,' --implicit-check-not='TOK_LPKEYWORD_PRIM(custom_single,' --implicit-check-not='TOK_LPKEYWORD_PRIM(read,' --implicit-check-not='TOK_LPKEYWORD_PRIM(probe,' --implicit-check-not='TOK_LPKEYWORD_PRIM(string_concat,'

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

def EmitOnlyCallOp : FIRRTLOp<"other.call"> {
  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_only` `(` operands `)`";
  let firrtlEmitOnly = 1;
  let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
  let firrtlFeature = "custom feature";
}

def EmitOnlyCastOp : FIRRTLOp<"other.cast"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let firrtlFormat = "$value";
  let firrtlEmitOnly = 1;
}

def EmitOnlySingleGroupOp : FIRRTLOp<"other.single"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_single` `(` operands `)`";
  let firrtlEmitOnly = 1;
}

def EmitOnlyUnaryOp : FIRRTLOp<"other.unary"> {
  let arguments = (ins IntType:$value);
  let results = (outs IntType:$result);
  let firrtlFormat = "`custom_unary` `(` $value `)`";
  let firrtlEmitOnly = 1;
}

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(EmitOnlyCallOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "custom feature"))) return;
// CHECK-NEXT: emitPrimExpr("custom_only", op);
// CHECK: void emitExpression(EmitOnlyCastOp op) {
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK: void emitExpression(EmitOnlySingleGroupOp op) {
// CHECK-NEXT: ps << "custom_single(";
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK-NEXT: ps << ")";
// CHECK: void emitExpression(EmitOnlyUnaryOp op) {
// CHECK-NEXT: ps << "custom_unary(";
// CHECK-NEXT: emitExpression(op.getValue());
// CHECK-NEXT: ps << ")";
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: EmitOnlyCallOp,
// CHECK: EmitOnlyCastOp,
// CHECK: EmitOnlySingleGroupOp,
// CHECK: EmitOnlyUnaryOp,
