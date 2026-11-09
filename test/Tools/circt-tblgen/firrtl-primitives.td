// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp --implicit-check-not=TagExtractOp --implicit-check-not=SizeOfIntrinsicOp --implicit-check-not=ElementwiseAndPrimOp --implicit-check-not=ElementwiseOrPrimOp --implicit-check-not=ElementwiseXorPrimOp --implicit-check-not=Mux2CellIntrinsicOp --implicit-check-not=Mux4CellIntrinsicOp --implicit-check-not=ListContainsOp --implicit-check-not=PropEqOp --implicit-check-not=RefResolveOp --implicit-check-not=RefSendOp --implicit-check-not=StringConcatOp --implicit-check-not=UnsafeDomainCastOp --implicit-check-not=ConstCastOp --implicit-check-not=RefCastOp --implicit-check-not=DomainAnyRefCastOp --implicit-check-not=UninferredResetCastOp

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK: TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(asReset, AsResetPrimOp, 1, 0, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(bool_and, BoolAndOp, 2, 0, FIRVersion(6, 0, 0), "boolean and")
// CHECK: TOK_LPKEYWORD_PRIM(integer_add, IntegerAddOp, 2, 0, FIRVersion(4, 0, 0), "integer_add")
// CHECK: TOK_LPKEYWORD_PRIM(integer_and, IntegerAndOp, 2, 0, missingSpecFIRVersion, "integer_and")
// CHECK: TOK_LPKEYWORD_PRIM(mux, MuxPrimOp, 3, 0, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(shl, ShlPrimOp, 1, 1, FIRVersion(0, 0, 0), "Base")
// CHECK: TOK_LPKEYWORD_PRIM(shr, ShrPrimOp, 1, 1, FIRVersion(0, 0, 0), "Base")
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(AddPrimOp op) {
// CHECK-NEXT: emitPrimExpr("add", op);
// CHECK: void emitExpression(BitsPrimOp op) {
// CHECK-NEXT: emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
// CHECK: void emitExpression(BoolAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "boolean and"))) return;
// CHECK-NEXT: emitPrimExpr("bool_and", op);
// CHECK: void emitExpression(IntegerAddOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_add"))) return;
// CHECK-NEXT: emitPrimExpr("integer_add", op);
// CHECK: void emitExpression(MuxPrimOp op) {
// CHECK-NEXT: emitPrimExpr("mux", op);
// CHECK: void emitExpression(ShlPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shl", op, op.getAmount());
// CHECK: void emitExpression(ShrPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shr", op, op.getAmount());
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: BitsPrimOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
