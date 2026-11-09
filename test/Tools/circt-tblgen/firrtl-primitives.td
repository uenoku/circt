// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s --implicit-check-not=CatPrimOp --implicit-check-not=TagExtractOp --implicit-check-not=SizeOfIntrinsicOp --implicit-check-not=ElementwiseAndPrimOp --implicit-check-not=ElementwiseOrPrimOp --implicit-check-not=ElementwiseXorPrimOp --implicit-check-not=Mux2CellIntrinsicOp --implicit-check-not=Mux4CellIntrinsicOp --implicit-check-not='TOK_LPKEYWORD_PRIM(unsafe_domain_cast,' --implicit-check-not='TOK_LPKEYWORD_PRIM(read,' --implicit-check-not='TOK_LPKEYWORD_PRIM(probe,' --implicit-check-not='TOK_LPKEYWORD_PRIM(string_concat,'

include "circt/Dialect/FIRRTL/FIRRTLExpressions.td"

// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_TOKENS
// CHECK: TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(asReset, AsResetPrimOp, 1, 0, FIRVersion(0, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(bool_and, BoolAndOp, 2, 0, FIRVersion(6, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(integer_add, IntegerAddOp, 2, 0, FIRVersion(4, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(integer_and, IntegerAndOp, 2, 0, missingSpecFIRVersion)
// CHECK: TOK_LPKEYWORD_PRIM(list_contains, ListContainsOp, 2, 0, missingSpecFIRVersion)
// CHECK: TOK_LPKEYWORD_PRIM(mux, MuxPrimOp, 3, 0, FIRVersion(0, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(prop_eq, PropEqOp, 2, 0, FIRVersion(6, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(shl, ShlPrimOp, 1, 1, FIRVersion(0, 0, 0))
// CHECK: TOK_LPKEYWORD_PRIM(shr, ShrPrimOp, 1, 1, FIRVersion(0, 0, 0))
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_EMITTERS
// CHECK: void emitExpression(AddPrimOp op) {
// CHECK-NEXT: emitPrimExpr("add", op);
// CHECK: void emitExpression(BitsPrimOp op) {
// CHECK-NEXT: emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
// CHECK: void emitExpression(BoolAndOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "bool_and"))) return;
// CHECK-NEXT: emitPrimExpr("bool_and", op);
// CHECK: void emitExpression(ConstCastOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK: void emitExpression(DomainAnyRefCastOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK: void emitExpression(IntegerAddOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(4, 0, 0), op, "integer_add"))) return;
// CHECK-NEXT: emitPrimExpr("integer_add", op);
// CHECK: void emitExpression(ListContainsOp op) {
// CHECK-NEXT: if (failed(requireVersion(missingSpecFIRVersion, op, "list_contains"))) return;
// CHECK-NEXT: emitPrimExpr("list_contains", op);
// CHECK: void emitExpression(MuxPrimOp op) {
// CHECK-NEXT: emitPrimExpr("mux", op);
// CHECK: void emitExpression(PropEqOp op) {
// CHECK-NEXT: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "prop_eq"))) return;
// CHECK-NEXT: emitPrimExpr("prop_eq", op);
// CHECK: void emitExpression(RefCastOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK: void emitExpression(RefResolveOp op) {
// CHECK-NEXT: ps << "read(";
// CHECK-NEXT: emitExpression(op.getRef());
// CHECK-NEXT: ps << ")";
// CHECK: void emitExpression(RefSendOp op) {
// CHECK-NEXT: ps << "probe(";
// CHECK-NEXT: emitExpression(op.getBase());
// CHECK-NEXT: ps << ")";
// CHECK: void emitExpression(ShlPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shl", op, op.getAmount());
// CHECK: void emitExpression(ShrPrimOp op) {
// CHECK-NEXT: emitPrimExpr("shr", op, op.getAmount());
// CHECK: void emitExpression(StringConcatOp op) {
// CHECK-NEXT: emitPrimExpr("string_concat", op);
// CHECK: void emitExpression(UninferredResetCastOp op) {
// CHECK-NEXT: emitExpression(op.getInput());
// CHECK: void emitExpression(UnsafeDomainCastOp op) {
// CHECK-NEXT: if (failed(requireVersion(nextFIRVersion, op, "unsafe_domain_cast"))) return;
// CHECK-NEXT: emitPrimExpr("unsafe_domain_cast", op);
// CHECK: #ifdef GET_FIRRTL_PRIMITIVE_DISPATCH
// CHECK: AddPrimOp,
// CHECK: BitsPrimOp,
// CHECK: ConstCastOp,
// CHECK: DomainAnyRefCastOp,
// CHECK: ListContainsOp,
// CHECK: PropEqOp,
// CHECK: RefCastOp,
// CHECK: RefResolveOp,
// CHECK: RefSendOp,
// CHECK: ShlPrimOp,
// CHECK: ShrPrimOp,
// CHECK: StringConcatOp,
// CHECK: UninferredResetCastOp,
// CHECK: UnsafeDomainCastOp,
