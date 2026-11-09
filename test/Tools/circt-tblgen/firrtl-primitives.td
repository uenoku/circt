// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp --implicit-check-not=TagExtractOp --implicit-check-not=SizeOfIntrinsicOp --implicit-check-not=ElementwiseAndPrimOp --implicit-check-not=ElementwiseOrPrimOp --implicit-check-not=ElementwiseXorPrimOp --implicit-check-not=Mux2CellIntrinsicOp --implicit-check-not=Mux4CellIntrinsicOp --implicit-check-not='TOK_LPKEYWORD_PRIM(list_contains,' --implicit-check-not='TOK_LPKEYWORD_PRIM(prop_eq,' --implicit-check-not='TOK_LPKEYWORD_PRIM(read,' --implicit-check-not='TOK_LPKEYWORD_PRIM(probe,' --implicit-check-not='TOK_LPKEYWORD_PRIM(string_concat,' --implicit-check-not='TOK_LPKEYWORD_PRIM(unsafe_domain_cast,'

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
// CHECK: void emitExpression(ListContainsOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "list contains"))) return;
// CHECK-NEXT: emitPrimExpr("list_contains", op);
// CHECK: void emitExpression(PropEqOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "property equality"))) return;
// CHECK-NEXT: emitPrimExpr("prop_eq", op);
// CHECK: void emitExpression(RefResolveOp op) {
// CHECK-NEXT: emitUnaryCall("read", op);
// CHECK: void emitExpression(RefSendOp op) {
// CHECK-NEXT: emitUnaryCall("probe", op);
// CHECK: void emitExpression(StringConcatOp op) {
// CHECK-NEXT: emitPrimExpr("string_concat", op);
// CHECK: void emitExpression(UnsafeDomainCastOp op) {
// CHECK-NEXT: if (failed(requireVersion(nextFIRVersion, op, "unsafe_domain_cast"))) return;
// CHECK-NEXT: emitUnboxedCall("unsafe_domain_cast", op);
// CHECK: void emitExpression(ConstCastOp op) {
// CHECK-NEXT: emitExpression(op->getOperand(0));
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: BitsPrimOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
// CHECK: ListContainsOp,
// CHECK: RefResolveOp,
// CHECK: ConstCastOp,
