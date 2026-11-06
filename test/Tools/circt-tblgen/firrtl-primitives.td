// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK: TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(asReset, AsResetPrimOp, 1, 0, missingSpecFIRVersion, "reset casts")
// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(bool_and, BoolAndOp, 2, 0, missingSpecFIRVersion, "Boolean and")
// CHECK: TOK_LPKEYWORD_PRIM(integer_and, IntegerAndOp, 2, 0, missingSpecFIRVersion, "Integers")
// CHECK: TOK_LPKEYWORD_PRIM(shl, ShlPrimOp, 1, 1, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(shr, ShrPrimOp, 1, 1, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(example, VersionedOp, 1, 1, missingSpecFIRVersion, "Import \"feature\"")
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(AddPrimOp op) {
// CHECK-NEXT: emitPrimExpr("add", op);
// CHECK: void emitExpression(BitsPrimOp op) {
// CHECK-NEXT: emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
// CHECK: void emitExpression(BoolAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "boolean and"))) return;
// CHECK-NEXT: emitPrimExpr("bool_and", op);
// CHECK: void emitExpression(IntegerAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "integer and"))) return;
// CHECK-NEXT: emitPrimExpr("integer_and", op);
// CHECK: void emitExpression(ShlPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shl", op, op.getAmount());
// CHECK: void emitExpression(VersionedOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "Export feature"))) return;
// CHECK-NEXT: emitPrimExpr("example", op, op.getExampleAmount());
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: BitsPrimOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
// CHECK: VersionedOp,

def VersionedOp : FIRRTLOp<"example">, FIRRTLPrimitive<"example"> {
  let arguments = (ins IntType:$input, I32Attr:$exampleAmount);
  let results = (outs IntType:$result);
  let firrtlImportVersion = FIRRTLUnspecifiedVersion;
  let firrtlImportFeature = "Import \"feature\"";
  let firrtlExportVersion = FIRRTLReleasedVersion<6, 0, 0>;
  let firrtlExportFeature = "Export feature";
}
