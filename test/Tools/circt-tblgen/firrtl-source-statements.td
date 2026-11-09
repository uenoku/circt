// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s | FileCheck %s
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.first
// RUN: circt-tblgen --firrtl-primitives %TABLEGEN_INC% %s > %t.second
// RUN: diff %t.first %t.second

include "circt/Dialect/FIRRTL/FIRRTLStatements.td"

// A different source spelling and version demonstrate that the generated
// statements are driven by the format and shared metadata, not op names.
def VersionedCallOp : FIRRTLOp<"other.call"> {
  let arguments = (ins ClockType:$clock, UInt1Type:$cond, I32Attr:$number,
                       I32Attr:$other, StrAttr:$label);
  let results = (outs);
  let firrtlFormat = "`other_call` `(` $clock `,` $cond `,` $number `,` $other `)` optionalName($label) info";
  let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
  let firrtlFeature = "other call";
}

def VersionedKeywordOp : FIRRTLOp<"other.keyword"> {
  let arguments = (ins);
  let results = (outs);
  let firrtlFormat = "`other_keyword` info";
  let firrtlVersion = FIRRTLNextVersion;
  let firrtlFeature = "other keyword";
}

// CHECK: #ifdef GET_FIRRTL_STATEMENT_TOKENS
// CHECK-NEXT: TOK_KEYWORD(skip)
// CHECK-NEXT: TOK_LPKEYWORD(stop)
// CHECK-NEXT: TOK_LPKEYWORD(other_call)
// CHECK-NEXT: TOK_KEYWORD(other_keyword)
// CHECK: #ifdef GET_FIRRTL_STATEMENT_PARSERS
// CHECK: case FIRToken::kw_skip: {
// CHECK: if (auto isExpr = parseExpWithLeadingKeyword(startTok)) return *isExpr;
// CHECK: if (parseOptionalInfo()) return failure();
// CHECK: SkipOp::create(builder);
// CHECK: case FIRToken::lp_stop: {
// CHECK: if (parseExp(operand0,
// CHECK: if (parseExp(operand1,
// CHECK: if (parseIntLit(integer0,
// CHECK: parseOptionalName(name) || parseOptionalInfo()) return failure();
// CHECK: StopOp::create(builder, operand0, operand1, builder.getI32IntegerAttr(integer0), name);
// CHECK: case FIRToken::lp_other_call: {
// CHECK: if (requireFeature(FIRVersion(6, 0, 0), "other call", startTok.getLoc())) return failure();
// CHECK: VersionedCallOp::create(builder, operand0, operand1, builder.getI32IntegerAttr(integer0), builder.getI32IntegerAttr(integer1), name);
// CHECK: case FIRToken::kw_other_keyword: {
// CHECK: if (requireFeature(nextFIRVersion, "other keyword", startTok.getLoc())) return failure();
// CHECK: #ifdef GET_FIRRTL_STATEMENT_EMITTERS
// CHECK: void emitStatement(SkipOp op) {
// CHECK: ps << "skip";
// CHECK: void emitStatement(StopOp op) {
// CHECK: ps << "stop(" << PP::ibox0;
// CHECK: emitExpression(op->getOperand(0));
// CHECK: emitExpression(op->getOperand(1));
// CHECK: ps.addAsString(op.getExitCode());
// CHECK: PPExtString(legalize(op.getNameAttr()));
// CHECK: void emitStatement(VersionedCallOp op) {
// CHECK: if (failed(requireVersion(FIRVersion(6, 0, 0), op, "other call"))) return;
// CHECK: ps << "other_call(" << PP::ibox0;
// CHECK: ps.addAsString(op.getNumber());
// CHECK: ps.addAsString(op.getOther());
// CHECK: PPExtString(legalize(op.getLabelAttr()));
// CHECK: void emitStatement(VersionedKeywordOp op) {
// CHECK: if (failed(requireVersion(nextFIRVersion, op, "other keyword"))) return;
// CHECK: #ifdef GET_FIRRTL_STATEMENT_DISPATCH
// CHECK-NEXT: SkipOp,
// CHECK-NEXT: StopOp,
// CHECK-NEXT: VersionedCallOp,
// CHECK-NEXT: VersionedKeywordOp,
