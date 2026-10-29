//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Liberty file import functionality.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportLiberty.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthAttributes.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "import-liberty"

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::comb;

namespace {

/// Liberty token types for lexical analysis
enum class LibertyTokenKind {
  // Literals
  Identifier,
  String,
  Number,

  // Punctuation
  LBrace, // {
  RBrace, // }
  LParen, // (
  RParen, // )
  Colon,  // :
  Semi,   // ;
  Comma,  // ,

  // Special
  EndOfFile,
  Error
};

StringRef stringifyTokenKind(LibertyTokenKind kind) {
  switch (kind) {
  case LibertyTokenKind::Identifier:
    return "identifier";
  case LibertyTokenKind::String:
    return "string";
  case LibertyTokenKind::Number:
    return "number";
  case LibertyTokenKind::LBrace:
    return "'{'";
  case LibertyTokenKind::RBrace:
    return "'}'";
  case LibertyTokenKind::LParen:
    return "'('";
  case LibertyTokenKind::RParen:
    return "')'";
  case LibertyTokenKind::Colon:
    return "':'";
  case LibertyTokenKind::Semi:
    return "';'";
  case LibertyTokenKind::Comma:
    return "','";

  case LibertyTokenKind::EndOfFile:
    return "end of file";
  case LibertyTokenKind::Error:
    return "error";
  }
  return "unknown";
}

struct LibertyToken {
  LibertyTokenKind kind;
  StringRef spelling;
  SMLoc location;

  LibertyToken(LibertyTokenKind kind, StringRef spelling, SMLoc location)
      : kind(kind), spelling(spelling), location(location) {}

  bool is(LibertyTokenKind k) const { return kind == k; }
};

class LibertyLexer {
public:
  LibertyLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
      : sourceMgr(sourceMgr), context(context),
        curBuffer(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
        curPtr(curBuffer.begin()) {}

  LibertyToken nextToken();
  LibertyToken peekToken();

  SMLoc getCurrentLoc() const { return SMLoc::getFromPointer(curPtr); }

  Location translateLocation(llvm::SMLoc loc) const {
    unsigned mainFileID = sourceMgr.getMainFileID();
    auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
    return FileLineColLoc::get(
        StringAttr::get(
            context,
            sourceMgr.getMemoryBuffer(mainFileID)->getBufferIdentifier()),
        lineAndColumn.first, lineAndColumn.second);
  }

  bool isAtEnd() const { return curPtr >= curBuffer.end(); }

private:
  const llvm::SourceMgr &sourceMgr;
  MLIRContext *context;
  StringRef curBuffer;
  const char *curPtr;

  void skipWhitespaceAndComments();
  LibertyToken lexIdentifier();
  LibertyToken lexString();
  LibertyToken lexNumber();
  bool isNumberStart() const;
  LibertyToken makeToken(LibertyTokenKind kind, const char *start);
};

/// Helper class to parse boolean expressions from Liberty function attributes.
///
/// This parser implements a recursive-descent parser for Liberty boolean
/// expressions with the following operator precedence (highest to lowest):
///   1. NOT (!, ')      - Unary negation (prefix or postfix)
///   2. AND (*, &)      - Conjunction
///   3. XOR (^)         - Exclusive OR
///   4. OR  (+, |)      - Disjunction
///
/// Parentheses can be used to override precedence. The grammar is:
///   OrExpr    -> XorExpr { ('+'|'|') XorExpr }
///   XorExpr   -> AndExpr { '^' AndExpr }
///   AndExpr   -> UnaryExpr { ('*'|'&') UnaryExpr }
///   UnaryExpr -> '!' UnaryExpr | '(' OrExpr ')' ['\''] | ID ['\'] | 0 | 1
class ExpressionParser {
public:
  ExpressionParser(LibertyLexer &lexer, OpBuilder &builder, StringRef expr,
                   const llvm::StringMap<Value> &values, SMLoc baseLoc);

  ParseResult parse(Value &result);

private:
  enum class TokenKind {
    ID,
    AND,
    OR,
    XOR,
    PREFIX_NOT,  // !
    POSTFIX_NOT, // '
    LPAREN,
    RPAREN,
    END
  };

  struct Token {
    TokenKind kind;
    StringRef spelling;
    SMLoc loc;
  };

  LibertyLexer &lexer;
  OpBuilder &builder;
  const llvm::StringMap<Value> &values;
  SMLoc baseLoc;
  const char *exprStart;
  SmallVector<Token> tokens;
  size_t pos = 0;
  Value trueVal = nullptr;

  SMLoc getLoc(const char *ptr);
  void tokenize(StringRef expr);
  Token peek() const;
  Token consume();
  Value createNot(Value val, SMLoc loc);
  Value createConstant(bool value, SMLoc loc);

  ParseResult parseOrExpr(Value &result);
  ParseResult parseXorExpr(Value &result);
  ParseResult parseAndExpr(Value &result);
  ParseResult parseUnaryExpr(Value &result);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) const;
};

//===----------------------------------------------------------------------===//
// ExpressionParser Implementation
//===----------------------------------------------------------------------===//

ExpressionParser::ExpressionParser(LibertyLexer &lexer, OpBuilder &builder,
                                   StringRef expr,
                                   const llvm::StringMap<Value> &values,
                                   SMLoc baseLoc)
    : lexer(lexer), builder(builder), values(values), baseLoc(baseLoc),
      exprStart(expr.begin()) {
  tokenize(expr);
}

ParseResult ExpressionParser::parse(Value &result) {
  return parseOrExpr(result);
}

SMLoc ExpressionParser::getLoc(const char *ptr) {
  size_t offset = ptr - exprStart;
  return SMLoc::getFromPointer(baseLoc.getPointer() + 1 + offset);
}

void ExpressionParser::tokenize(StringRef expr) {
  const char *ptr = expr.begin();
  const char *end = expr.end();

  while (ptr < end) {
    // Skip whitespace
    if (isspace(*ptr)) {
      ++ptr;
      continue;
    }

    SMLoc loc = getLoc(ptr);

    // Identifier
    if (isalnum(*ptr) || *ptr == '_') {
      const char *start = ptr;
      while (ptr < end && (isalnum(*ptr) || *ptr == '_'))
        ++ptr;
      tokens.push_back({TokenKind::ID, StringRef(start, ptr - start), loc});
      continue;
    }

    // Operators and punctuation
    switch (*ptr) {
    case '*':
    case '&':
      tokens.push_back({TokenKind::AND, StringRef(ptr, 1), loc});
      break;
    case '+':
    case '|':
      tokens.push_back({TokenKind::OR, StringRef(ptr, 1), loc});
      break;
    case '^':
      tokens.push_back({TokenKind::XOR, StringRef(ptr, 1), loc});
      break;
    case '!':
      tokens.push_back({TokenKind::PREFIX_NOT, StringRef(ptr, 1), loc});
      break;
    case '\'':
      tokens.push_back({TokenKind::POSTFIX_NOT, StringRef(ptr, 1), loc});
      break;
    case '(':
      tokens.push_back({TokenKind::LPAREN, StringRef(ptr, 1), loc});
      break;
    case ')':
      tokens.push_back({TokenKind::RPAREN, StringRef(ptr, 1), loc});
      break;
    }
    ++ptr;
  }

  tokens.push_back({TokenKind::END, "", getLoc(ptr)});
}

ExpressionParser::Token ExpressionParser::peek() const { return tokens[pos]; }

ExpressionParser::Token ExpressionParser::consume() { return tokens[pos++]; }

Value ExpressionParser::createNot(Value val, SMLoc loc) {
  if (!trueVal)
    trueVal = createConstant(true, loc);
  return comb::XorOp::create(builder, lexer.translateLocation(loc), val,
                             trueVal);
}

Value ExpressionParser::createConstant(bool value, SMLoc loc) {
  return hw::ConstantOp::create(builder, lexer.translateLocation(loc),
                                builder.getI1Type(), value ? 1 : 0);
}

/// Parse OR expressions with lowest precedence.
/// OrExpr -> XorExpr { ('+'|'|') XorExpr }
/// This implements left-associative parsing: A + B + C becomes (A + B) + C
ParseResult ExpressionParser::parseOrExpr(Value &result) {
  Value lhs;
  if (parseXorExpr(lhs))
    return failure();
  while (peek().kind == TokenKind::OR) {
    auto loc = consume().loc;
    Value rhs;
    if (parseXorExpr(rhs))
      return failure();
    lhs = comb::OrOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
  }
  result = lhs;
  return success();
}

/// Parse XOR expressions with medium precedence.
/// XorExpr -> AndExpr { '^' AndExpr }
/// This implements left-associative parsing: A ^ B ^ C becomes (A ^ B) ^ C
ParseResult ExpressionParser::parseXorExpr(Value &result) {
  Value lhs;
  if (parseAndExpr(lhs))
    return failure();
  while (peek().kind == TokenKind::XOR) {
    auto loc = consume().loc;
    Value rhs;
    if (parseAndExpr(rhs))
      return failure();
    lhs = comb::XorOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
  }
  result = lhs;
  return success();
}

/// Parse AND expressions with highest precedence.
/// AndExpr -> UnaryExpr { ('*'|'&'|implicit) UnaryExpr }
/// This implements left-associative parsing: A * B * C becomes (A * B) * C
/// Space between expressions is treated as implicit AND.
ParseResult ExpressionParser::parseAndExpr(Value &result) {
  Value lhs;
  if (parseUnaryExpr(lhs))
    return failure();
  while (peek().kind == TokenKind::AND ||
         (peek().kind == TokenKind::ID || peek().kind == TokenKind::LPAREN ||
          peek().kind == TokenKind::PREFIX_NOT)) {
    auto loc = peek().loc;
    if (peek().kind == TokenKind::AND)
      consume();
    Value rhs;
    if (parseUnaryExpr(rhs))
      return failure();
    lhs = comb::AndOp::create(builder, lexer.translateLocation(loc), lhs, rhs);
  }
  result = lhs;
  return success();
}

/// Parse unary expressions and primary expressions.
/// UnaryExpr -> '!' UnaryExpr | '(' OrExpr ')' ['\''] | ID ['\']
/// Handles prefix NOT (!), parenthesized expressions, and identifiers.
/// Postfix NOT (') is only allowed after expressions and identifiers.
ParseResult ExpressionParser::parseUnaryExpr(Value &result) {
  // Prefix NOT (!)
  if (peek().kind == TokenKind::PREFIX_NOT) {
    auto loc = consume().loc;
    Value val;
    if (parseUnaryExpr(val))
      return failure();
    result = createNot(val, loc);
    return success();
  }

  // Parenthesized expression
  if (peek().kind == TokenKind::LPAREN) {
    consume();
    Value val;
    if (parseOrExpr(val))
      return failure();
    if (peek().kind != TokenKind::RPAREN)
      return failure();
    consume();
    // Postfix NOT (')
    if (peek().kind == TokenKind::POSTFIX_NOT) {
      auto loc = consume().loc;
      val = createNot(val, loc);
    }
    result = val;
    return success();
  }

  // Identifier
  if (peek().kind == TokenKind::ID) {
    auto tok = consume();
    StringRef name = tok.spelling;
    if (name == "0" || name == "1") {
      Value val = createConstant(name == "1", tok.loc);
      if (peek().kind == TokenKind::POSTFIX_NOT) {
        auto loc = consume().loc;
        val = createNot(val, loc);
      }
      result = val;
      return success();
    }
    auto it = values.find(name);
    if (it == values.end())
      return emitError(tok.loc, "variable not found");
    Value val = it->second;
    // Postfix NOT (')
    if (peek().kind == TokenKind::POSTFIX_NOT) {
      auto loc = consume().loc;
      val = createNot(val, loc);
    }
    result = val;
    return success();
  }

  return emitError(peek().loc, "expected expression");
}

InFlightDiagnostic ExpressionParser::emitError(llvm::SMLoc loc,
                                               const Twine &message) const {
  return mlir::emitError(lexer.translateLocation(loc), message);
}

// Parsed result of a Liberty group
struct LibertyGroup {
  StringRef name;
  SMLoc loc;
  SmallVector<Attribute> args;
  struct AttrEntry {
    StringRef name;
    Attribute value;
    SMLoc loc;
    AttrEntry(StringRef name, Attribute value, SMLoc loc)
        : name(name), value(value), loc(loc) {}
  };
  SmallVector<AttrEntry> attrs;
  SmallVector<std::unique_ptr<LibertyGroup>> subGroups;

  // Helper to find an attribute by name
  std::pair<Attribute, SMLoc> getAttribute(StringRef name) const {
    for (const auto &attr : attrs)
      if (attr.name == name)
        return {attr.value, attr.loc};
    return {};
  }

  void eraseSubGroup(llvm::function_ref<bool(const LibertyGroup &)> pred) {
    subGroups.erase(
        std::remove_if(subGroups.begin(), subGroups.end(),
                       [pred](const auto &group) { return pred(*group); }),
        subGroups.end());
  }

  LogicalResult checkArgs(
      size_t n,
      llvm::function_ref<LogicalResult(size_t idx, Attribute)> pred) const {
    if (args.size() != n)
      return failure();
    for (size_t i = 0; i < n; ++i)
      if (failed(pred(i, args[i])))
        return failure();
    return success();
  }
};

class LibertyParser {
public:
  LibertyParser(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                ModuleOp module)
      : lexer(sourceMgr, context), module(module),
        builder(module.getBodyRegion()) {}

  ParseResult parse();

private:
  LibertyLexer lexer;
  ModuleOp module;
  OpBuilder builder;

  // Specific group parsers
  ParseResult lowerLibrary(LibertyGroup &libertyLib);
  ParseResult parseGroupBody(LibertyGroup &group);
  ParseResult parseStatement(LibertyGroup &parent);

  // Lowering methods
  ParseResult
  lowerCell(const LibertyGroup &group, StringAttr &cellNameAttr,
            const llvm::StringMap<const LibertyGroup *> &tableTemplates);

  ParseResult
  parseNLDMTable(const LibertyGroup &group,
                 const llvm::StringMap<const LibertyGroup *> &tableTemplates,
                 synth::NLDMTableAttr &table);

  LogicalResult
  buildTimingArcs(const LibertyGroup &pinGroup,
                  const llvm::StringMap<const LibertyGroup *> &tableTemplates,
                  SmallVector<Attribute> &timingArcs);

  LogicalResult
  collectPorts(const LibertyGroup &cellGroup,
               const llvm::StringMap<const LibertyGroup *> &tableTemplates,
               SmallVector<hw::PortInfo> &ports,
               SmallVector<const LibertyGroup *> &pinGroups,
               SmallVector<Attribute> &timingArcs,
               size_t &numOutputPinMissingFunction, bool &hasThreeStateOutput);

  LogicalResult buildCellLogic(hw::HWModuleOp moduleOp,
                               ArrayRef<hw::PortInfo> ports,
                               ArrayRef<const LibertyGroup *> pinGroups);

  LogicalResult
  buildSequentialCellLogic(hw::HWModuleOp moduleOp,
                           ArrayRef<hw::PortInfo> ports,
                           ArrayRef<const LibertyGroup *> pinGroups,
                           const LibertyGroup &seqGroup, bool isLatch);

  LogicalResult
  buildClockGateCellLogic(hw::HWModuleOp moduleOp, ArrayRef<hw::PortInfo> ports,
                          ArrayRef<const LibertyGroup *> pinGroups,
                          const LibertyGroup &cellGroup);

  // A Liberty signal reference for clocks and asynchronous controls must be
  // a bare pin or an inverted pin (`[!]name`). Returns the value and whether
  // it was inverted.
  FailureOr<std::pair<Value, bool>>
  resolvePinRef(StringRef expr, SMLoc loc,
                const llvm::StringMap<Value> &values);

  // Normalize to active-high: inverted references become a `comb` inversion.
  Value activeHigh(Value val, bool inverted, Location loc);

  LogicalResult parseLibertyFloatList(StringRef value, SMLoc loc,
                                      SmallVectorImpl<double> &result) const;

  //===--------------------------------------------------------------------===//
  // Parser for Subgroup of "cell" group.
  //===--------------------------------------------------------------------===//

  Attribute convertGroupToAttr(const LibertyGroup &group);

  // Attribute parsing
  ParseResult parseAttribute(Attribute &result);

  // Expression parsing
  ParseResult parseExpression(Value &result, StringRef expr,
                              const llvm::StringMap<Value> &values, SMLoc loc);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }

  InFlightDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message) const {
    return mlir::emitWarning(lexer.translateLocation(loc), message);
  }

  ParseResult consume(LibertyTokenKind kind, const Twine &msg) {
    if (lexer.nextToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(), msg);
  }

  ParseResult consumeUntil(LibertyTokenKind kind) {
    while (!lexer.peekToken().is(kind) &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile)
      lexer.nextToken();
    if (lexer.peekToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(),
                     " expected " + stringifyTokenKind(kind));
  }

  ParseResult consume(LibertyTokenKind kind) {
    if (lexer.nextToken().is(kind))
      return success();
    return emitError(lexer.getCurrentLoc(),
                     " expected " + stringifyTokenKind(kind));
  }

  ParseResult expect(LibertyTokenKind kind) {
    if (!lexer.peekToken().is(kind))
      return emitError(lexer.getCurrentLoc(),
                       " expected " + stringifyTokenKind(kind));

    return success();
  }

  StringRef getTokenSpelling(const LibertyToken &token) {
    StringRef str = token.spelling;
    if (token.kind == LibertyTokenKind::String)
      str = str.drop_front().drop_back();
    return str;
  }
};

//===----------------------------------------------------------------------===//
// LibertyLexer Implementation
//===----------------------------------------------------------------------===//

void LibertyLexer::skipWhitespaceAndComments() {
  while (curPtr < curBuffer.end()) {
    if (isspace(*curPtr)) {
      ++curPtr;
      continue;
    }

    // Comments
    if (*curPtr == '/' && curPtr + 1 < curBuffer.end()) {
      if (*(curPtr + 1) == '*') { // /* ... */
        curPtr += 2;
        while (curPtr + 1 < curBuffer.end() &&
               (*curPtr != '*' || *(curPtr + 1) != '/'))
          ++curPtr;
        if (curPtr + 1 < curBuffer.end())
          curPtr += 2;
        continue;
      }
      if (*(curPtr + 1) == '/') { // // ...
        while (curPtr < curBuffer.end() && *curPtr != '\n')
          ++curPtr;
        continue;
      }
    }

    // Backslash newline continuation
    if (*curPtr == '\\' && curPtr + 1 < curBuffer.end() &&
        *(curPtr + 1) == '\n') {
      curPtr += 2;
      continue;
    }

    break;
  }
}

LibertyToken LibertyLexer::lexIdentifier() {
  const char *start = curPtr;
  while (curPtr < curBuffer.end() &&
         (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '.' ||
          *curPtr == '[' || *curPtr == ']'))
    ++curPtr;
  return makeToken(LibertyTokenKind::Identifier, start);
}

LibertyToken LibertyLexer::lexString() {
  const char *start = curPtr;
  ++curPtr; // skip opening quote
  while (curPtr < curBuffer.end() && *curPtr != '"') {
    if (*curPtr == '\\' && curPtr + 1 < curBuffer.end())
      ++curPtr; // skip escaped char
    ++curPtr;
  }
  if (curPtr < curBuffer.end())
    ++curPtr; // skip closing quote
  return makeToken(LibertyTokenKind::String, start);
}

LibertyToken LibertyLexer::lexNumber() {
  const char *start = curPtr;
  if (curPtr < curBuffer.end() && (*curPtr == '+' || *curPtr == '-'))
    ++curPtr;

  while (curPtr < curBuffer.end() && isdigit(*curPtr))
    ++curPtr;

  if (curPtr < curBuffer.end() && *curPtr == '.') {
    ++curPtr;
    while (curPtr < curBuffer.end() && isdigit(*curPtr))
      ++curPtr;
    if (curPtr < curBuffer.end() && *curPtr == '.') {
      mlir::emitError(translateLocation(SMLoc::getFromPointer(curPtr)),
                      "multiple decimal points in number");
      return makeToken(LibertyTokenKind::Error, start);
    }
  }

  // Liberty files commonly use scientific notation for capacitance and
  // timing values (for example, 1.42e-05).  Consume the exponent as part of
  // the number so that the sign in the exponent is not treated as a separate
  // token.
  if (curPtr < curBuffer.end() && (*curPtr == 'e' || *curPtr == 'E')) {
    const char *exponent = curPtr++;
    if (curPtr < curBuffer.end() && (*curPtr == '+' || *curPtr == '-'))
      ++curPtr;
    const char *exponentDigits = curPtr;
    while (curPtr < curBuffer.end() && isdigit(*curPtr))
      ++curPtr;
    if (curPtr == exponentDigits) {
      mlir::emitError(translateLocation(SMLoc::getFromPointer(exponent)),
                      "expected digits in number exponent");
      return makeToken(LibertyTokenKind::Error, start);
    }
  }

  // Some Liberty producers attach a unit suffix to numeric values (for
  // example, 1.0ns).  The importer stores numeric attributes in the library's
  // native units, so retain the numeric value and consume the suffix here to
  // keep it from being parsed as a second statement token.
  while (curPtr < curBuffer.end() && (isalpha(*curPtr) || *curPtr == '_'))
    ++curPtr;

  return makeToken(LibertyTokenKind::Number, start);
}

bool LibertyLexer::isNumberStart() const {
  if (curPtr >= curBuffer.end())
    return false;

  if (isdigit(*curPtr))
    return true;
  if (*curPtr == '.' && curPtr + 1 < curBuffer.end() && isdigit(*(curPtr + 1)))
    return true;
  if (*curPtr != '+' && *curPtr != '-')
    return false;

  const char *next = curPtr + 1;
  return next < curBuffer.end() &&
         (isdigit(*next) ||
          (*next == '.' && next + 1 < curBuffer.end() && isdigit(*(next + 1))));
}

LibertyToken LibertyLexer::makeToken(LibertyTokenKind kind, const char *start) {
  return LibertyToken(kind, StringRef(start, curPtr - start),
                      SMLoc::getFromPointer(start));
}

LibertyToken LibertyLexer::nextToken() {
  skipWhitespaceAndComments();

  if (curPtr >= curBuffer.end())
    return makeToken(LibertyTokenKind::EndOfFile, curPtr);

  const char *start = curPtr;
  char c = *curPtr;

  if (isalpha(c) || c == '_')
    return lexIdentifier();

  if (isNumberStart())
    return lexNumber();

  if (c == '"')
    return lexString();

  ++curPtr;
  switch (c) {
  case '{':
    return makeToken(LibertyTokenKind::LBrace, start);
  case '}':
    return makeToken(LibertyTokenKind::RBrace, start);
  case '(':
    return makeToken(LibertyTokenKind::LParen, start);
  case ')':
    return makeToken(LibertyTokenKind::RParen, start);
  case ':':
    return makeToken(LibertyTokenKind::Colon, start);
  case ';':
    return makeToken(LibertyTokenKind::Semi, start);
  case ',':
    return makeToken(LibertyTokenKind::Comma, start);

  default:
    return makeToken(LibertyTokenKind::Error, start);
  }
}

LibertyToken LibertyLexer::peekToken() {
  const char *savedPtr = curPtr;
  LibertyToken token = nextToken();
  curPtr = savedPtr;
  return token;
}

//===----------------------------------------------------------------------===//
// LibertyParser Implementation
//===----------------------------------------------------------------------===//

ParseResult LibertyParser::parse() {
  // Parse all top-level groups first. Shipped PDKs contain file-splice
  // artifacts (seen in asap7sc6t SIMPLE: a `cell` after the library's
  // closing brace plus a stray `}`); adopt orphan cells into the preceding
  // library and skip stray braces with a warning instead of failing the
  // whole import. Anything else unexpected stays a hard error.
  SmallVector<std::unique_ptr<LibertyGroup>> libraries;
  while (lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::RBrace) {
      emitWarning(token.location, "stray '}' at top level; skipping");
      lexer.nextToken();
      continue;
    }
    if (token.kind != LibertyTokenKind::Identifier)
      return emitError(token.location, "expected `library` keyword");
    if (token.spelling == "library") {
      lexer.nextToken();
      auto lib = std::make_unique<LibertyGroup>();
      lib->name = "library";
      lib->loc = token.location;
      if (parseGroupBody(*lib))
        return failure();
      libraries.push_back(std::move(lib));
      continue;
    }
    if (token.spelling == "cell") {
      if (libraries.empty())
        return emitError(token.location, "expected `library` keyword");
      emitWarning(token.location, "top-level `cell` outside any `library` "
                                  "group; adopting into the preceding "
                                  "library");
      lexer.nextToken();
      auto orphan = std::make_unique<LibertyGroup>();
      orphan->name = "cell";
      orphan->loc = token.location;
      if (parseGroupBody(*orphan))
        return failure();
      libraries.back()->subGroups.push_back(std::move(orphan));
      continue;
    }
    return emitError(token.location, "expected `library` keyword");
  }
  for (auto &lib : libraries)
    if (lowerLibrary(*lib))
      return failure();
  return success();
}

ParseResult LibertyParser::lowerLibrary(LibertyGroup &libertyLib) {

  // First Pass: collect lu_table_template groups by name so they can be
  // referenced during NLDM table parsing in lowerCell. Liberty tables
  // reference templates by name for their index axes e.g.
  // cell_rise(delay_template_7x7) where the template defines index_1/index_2.

  llvm::StringMap<const LibertyGroup *> tableTemplates;
  for (const auto &stmt : libertyLib.subGroups)
    if (stmt->name == "lu_table_template")
      if (!stmt->args.empty())
        if (auto name = dyn_cast<StringAttr>(stmt->args[0]))
          tableTemplates[name.getValue()] = stmt.get();

  // Second Pass: lower cells, passing the template map through for index
  // lookup.
  DenseSet<StringAttr> seenCells;
  for (auto &stmt : libertyLib.subGroups) {
    if (stmt->name != "cell")
      continue;
    StringAttr cellNameAttr;
    if (lowerCell(*stmt, cellNameAttr, tableTemplates))
      return failure();
    if (!seenCells.insert(cellNameAttr).second)
      return emitError(stmt->loc, "redefinition of cell '" +
                                      cellNameAttr.getValue() + "'");
  }

  libertyLib.eraseSubGroup(
      [](const LibertyGroup &group) { return group.name == "cell"; });
  auto attr = convertGroupToAttr(libertyLib);
  module->setAttr("synth.liberty.library", attr);
  return success();
}

// Parse a template like: lu_table_template (delay_template_6x6) { variable_1:
// total_output_net_capacitance; variable_2: input_net_transition; }
Attribute LibertyParser::convertGroupToAttr(const LibertyGroup &group) {
  SmallVector<NamedAttribute> attrs;
  if (!group.args.empty())
    attrs.push_back(
        builder.getNamedAttr("args", builder.getArrayAttr(group.args)));

  for (const auto &attr : group.attrs)
    attrs.push_back(builder.getNamedAttr(attr.name, attr.value));

  llvm::StringMap<SmallVector<Attribute>> subGroups;
  for (const auto &sub : group.subGroups)
    subGroups[sub->name].push_back(convertGroupToAttr(*sub));

  for (auto &it : subGroups)
    attrs.push_back(
        builder.getNamedAttr(it.getKey(), builder.getArrayAttr(it.getValue())));

  return builder.getDictionaryAttr(attrs);
}

LogicalResult
LibertyParser::parseLibertyFloatList(StringRef value, SMLoc loc,
                                     SmallVectorImpl<double> &result) const {
  SmallVector<StringRef> parts;
  value.split(parts, ',');
  for (auto part : parts) {
    part = part.trim();
    if (part.empty())
      continue;
    double val;
    if (part.getAsDouble(val))
      return emitError(loc, "expected floating point value in NLDM table");
    result.push_back(val);
  }
  return success();
}

ParseResult LibertyParser::parseNLDMTable(
    const LibertyGroup &group,
    const llvm::StringMap<const LibertyGroup *> &tableTemplates,
    synth::NLDMTableAttr &table) {
  SmallVector<double> index1, index2, values;

  auto parseList = [&](const LibertyGroup &source, StringRef name,
                       SmallVectorImpl<double> &result) -> ParseResult {
    if (auto [attr, loc] = source.getAttribute(name); attr) {
      auto stringAttr = dyn_cast<StringAttr>(attr);
      if (!stringAttr)
        return emitError(loc, "expected string value for NLDM table entry");
      if (failed(parseLibertyFloatList(stringAttr.getValue(), loc, result)))
        return failure();
      return success();
    }

    for (const auto &subGroup : source.subGroups) {
      if (subGroup->name != name)
        continue;
      if (subGroup->args.empty())
        return emitError(subGroup->loc, "NLDM table entry missing value");
      for (auto arg : subGroup->args) {
        auto stringAttr = dyn_cast<StringAttr>(arg);
        if (!stringAttr)
          return emitError(subGroup->loc,
                           "expected string value for NLDM table entry");
        if (failed(parseLibertyFloatList(stringAttr.getValue(), subGroup->loc,
                                         result)))
          return failure();
      }
      return success();
    }

    return success();
  };

  // Table-local indices override template indices when present.
  if (parseList(group, "index_1", index1) ||
      parseList(group, "index_2", index2) || parseList(group, "values", values))
    return failure();

  bool isScalar = false;
  if (!group.args.empty()) {
    auto templateName = dyn_cast<StringAttr>(group.args[0]);
    if (!templateName)
      return emitError(group.loc, "NLDM table template name must be a string");

    isScalar = templateName.getValue() == "scalar";
    if (!isScalar) {
      auto it = tableTemplates.find(templateName.getValue());
      if (it == tableTemplates.end())
        return emitError(group.loc, "unknown NLDM table template '")
               << templateName.getValue() << "'";

      if (index1.empty() && parseList(*it->second, "index_1", index1))
        return failure();
      if (index2.empty() && parseList(*it->second, "index_2", index2))
        return failure();
    }
  }

  // Liberty's scalar template represents a constant value rather than a
  // lookup table.  NLDMTableAttr is uniformly 2D, so model it as a 1x1 table.
  if (isScalar) {
    if (index1.empty())
      index1.push_back(0.0);
    if (index2.empty())
      index2.push_back(0.0);
  }

  if (index1.empty())
    return emitError(group.loc, "NLDM table missing index_1");
  if (index2.empty())
    return emitError(group.loc, "NLDM table missing index_2");
  if (values.empty())
    return emitError(group.loc, "NLDM table missing values");

  if (values.size() != index1.size() * index2.size()) {
    return emitError(group.loc, "NLDM table has ")
           << values.size() << " values, expected "
           << index1.size() * index2.size() << " for " << index1.size() << " x "
           << index2.size() << " indices";
  }

  table = synth::NLDMTableAttr::get(
      builder.getContext(),
      DenseF64ArrayAttr::get(builder.getContext(), index1),
      DenseF64ArrayAttr::get(builder.getContext(), index2),
      DenseF64ArrayAttr::get(builder.getContext(), values));
  return success();
}

static bool isNLDMTableGroup(StringRef name) {
  return name == "cell_rise" || name == "cell_fall" ||
         name == "rise_transition" || name == "fall_transition";
}

LogicalResult LibertyParser::buildTimingArcs(
    const LibertyGroup &pinGroup,
    const llvm::StringMap<const LibertyGroup *> &tableTemplates,
    SmallVector<Attribute> &timingArcs) {
  auto pinName = cast<StringAttr>(pinGroup.args[0]);

  for (const auto &child : pinGroup.subGroups) {
    if (child->name != "timing")
      continue;

    // Constraint, recovery/removal, and min-pulse-width timing groups do not
    // describe an NLDM delay arc.  They are still valid Liberty, but are not
    // represented by the importer's current timing attribute.
    bool hasNLDMTable = llvm::any_of(child->subGroups, [](const auto &table) {
      return isNLDMTableGroup(table->name);
    });
    if (!hasNLDMTable)
      continue;

    // Sequential constraint arcs can contain only the transition direction
    // that applies to the constraint (for example, cell_fall and
    // fall_transition for a clear arc).  NLDMTimingArcAttr currently models a
    // complete delay arc, so leave these typed partial arcs in the generic
    // Liberty representation instead of rejecting an otherwise valid file.
    bool hasAllNLDMTables = llvm::all_of(
        ArrayRef<StringRef>{"cell_rise", "cell_fall", "rise_transition",
                            "fall_transition"},
        [&](StringRef name) {
          return llvm::any_of(child->subGroups, [&](const auto &table) {
            return table->name == name;
          });
        });
    if (!hasAllNLDMTables && child->getAttribute("timing_type").first)
      continue;

    auto relatedPin =
        dyn_cast_or_null<StringAttr>(child->getAttribute("related_pin").first);
    if (!relatedPin)
      return emitError(child->loc, "timing related_pin must be a string");

    // timing_sense is optional for combinational arcs.  Liberty defaults it to
    // non-unate when it is omitted.
    auto [timingSenseAttr, timingSenseLoc] =
        child->getAttribute("timing_sense");
    StringAttr timingSense = dyn_cast_or_null<StringAttr>(timingSenseAttr);
    if (!timingSenseAttr)
      timingSense = builder.getStringAttr("non_unate");
    else if (!timingSense)
      return emitError(timingSenseLoc, "timing_sense must be a string");

    synth::NLDMTableAttr cellRise, cellFall, riseTransition, fallTransition;
    for (const auto &table : child->subGroups) {
      if (!isNLDMTableGroup(table->name))
        continue;

      synth::NLDMTableAttr parsedTable;
      if (parseNLDMTable(*table, tableTemplates, parsedTable))
        return failure();

      if (table->name == "cell_rise")
        cellRise = parsedTable;
      else if (table->name == "cell_fall")
        cellFall = parsedTable;
      else if (table->name == "rise_transition")
        riseTransition = parsedTable;
      else if (table->name == "fall_transition")
        fallTransition = parsedTable;
    }

    if (!cellRise || !cellFall || !riseTransition || !fallTransition)
      return emitError(child->loc,
                       "NLDM timing arc requires cell_rise, cell_fall, "
                       "rise_transition, and fall_transition tables");

    timingArcs.push_back(synth::NLDMTimingArcAttr::get(
        builder.getContext(), pinName, relatedPin, timingSense, cellRise,
        cellFall, riseTransition, fallTransition));
  }
  return success();
}

LogicalResult LibertyParser::collectPorts(
    const LibertyGroup &cellGroup,
    const llvm::StringMap<const LibertyGroup *> &tableTemplates,
    SmallVector<hw::PortInfo> &ports,
    SmallVector<const LibertyGroup *> &pinGroups,
    SmallVector<Attribute> &timingArcs, size_t &numOutputPinMissingFunction,
    bool &hasThreeStateOutput) {

  llvm::DenseSet<StringAttr> seenPins;

  // Gather ports and count output pins without "function" attribute.
  for (const auto &sub : cellGroup.subGroups) {
    if (sub->name != "pin")
      continue;

    if (sub->args.empty())
      return emitError(sub->loc, "pin missing name");

    StringAttr pinName = dyn_cast<StringAttr>(sub->args[0]);
    if (!seenPins.insert(pinName).second)
      return emitError(sub->loc,
                       "redefinition of pin '" + pinName.getValue() + "'");

    std::optional<hw::ModulePort::Direction> dir;
    bool internalPin = false;
    SmallVector<NamedAttribute> pinAttrs;
    bool functionExist = false;
    bool threeStateExist = false;

    // Track if this pin has a "function" attribute (only relevant for outputs).
    bool isInOut = false;
    for (const auto &attr : sub->attrs) {
      if (attr.name == "direction") {
        auto val = dyn_cast<StringAttr>(attr.value);
        if (!val)
          return emitError(sub->loc, "pin direction must be a string");
        if (val.getValue() == "input")
          dir = hw::ModulePort::Direction::Input;
        else if (val.getValue() == "output")
          dir = hw::ModulePort::Direction::Output;
        else if (val.getValue() == "inout") {
          dir = hw::ModulePort::Direction::Input;
          isInOut = true;
        } else if (val.getValue() == "internal")
          internalPin = true;
        else
          return emitError(sub->loc,
                           "pin direction must be input, output, or inout");
        continue;
      }
      if (attr.name == "function")
        functionExist = true;
      if (attr.name == "three_state")
        threeStateExist = true;
      pinAttrs.push_back(builder.getNamedAttr(attr.name, attr.value));
    }

    // Liberty uses internal pins to describe implementation nodes, which are
    // not part of the cell interface and therefore cannot be HW module ports.
    if (internalPin)
      continue;

    if (!dir)
      return emitError(sub->loc, "pin direction must be specified");

    pinGroups.push_back(sub.get());

    if (dir == hw::ModulePort::Direction::Output && !functionExist)
      ++numOutputPinMissingFunction;
    // A three-state output is not driven unconditionally, so the cell cannot
    // lower to pure combinational logic even when it has a `function`.
    if (dir == hw::ModulePort::Direction::Output && threeStateExist)
      hasThreeStateOutput = true;

    // Build timing arcs for this pin's timing subgroups.
    if (failed(buildTimingArcs(*sub, tableTemplates, timingArcs)))
      return failure();

    // Collect non-timing subgroups as generic attrs
    llvm::StringMap<SmallVector<Attribute>> subGroupAttrs;
    for (const auto &child : sub->subGroups)
      if (child->name != "timing")
        subGroupAttrs[child->name].push_back(convertGroupToAttr(*child));

    for (auto &it : subGroupAttrs)
      pinAttrs.push_back(builder.getNamedAttr(
          it.getKey(), builder.getArrayAttr(it.getValue())));

    auto libertyAttrs = builder.getDictionaryAttr(pinAttrs);
    auto attrs = builder.getDictionaryAttr(
        builder.getNamedAttr("synth.liberty.pin", libertyAttrs));
    mlir::Type portType = builder.getI1Type();
    if (isInOut)
      portType = hw::InOutType::get(portType);

    hw::PortInfo port;
    port.name = pinName;
    port.type = portType;
    port.dir = *dir;
    port.attrs = attrs;
    ports.push_back(port);
  }
  return success();
}

FailureOr<std::pair<Value, bool>>
LibertyParser::resolvePinRef(StringRef expr, SMLoc loc,
                             const llvm::StringMap<Value> &values) {
  StringRef name = expr.trim();
  bool inverted = false;
  if (name.starts_with("!")) {
    inverted = true;
    name = name.drop_front().trim();
  }
  if (name.empty() || llvm::any_of(name, [](char c) {
        return !isalnum(c) && c != '_' && c != '[' && c != ']' && c != '.';
      }))
    return emitError(loc, "unsupported sequential signal expression '")
           << expr << "': expected a pin name";
  auto it = values.find(name);
  if (it == values.end())
    return emitError(loc, "unknown pin referenced by sequential cell: '")
           << name << "'";
  return std::make_pair(it->second, inverted);
}

Value LibertyParser::activeHigh(Value val, bool inverted, Location loc) {
  if (!inverted)
    return val;
  auto trueVal = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  return comb::XorOp::create(builder, loc, val, trueVal);
}

LogicalResult LibertyParser::buildSequentialCellLogic(
    hw::HWModuleOp moduleOp, ArrayRef<hw::PortInfo> ports,
    ArrayRef<const LibertyGroup *> pinGroups, const LibertyGroup &seqGroup,
    bool isLatch) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  auto loc = lexer.translateLocation(seqGroup.loc);

  llvm::StringMap<Value> portValues;
  for (const auto &port : ports)
    if (port.dir == hw::ModulePort::Direction::Input)
      portValues[port.name.getValue()] =
          moduleOp.getBodyBlock()->getArgument(port.argNum);

  auto getStringField = [&](StringRef field, bool required, StringAttr &result,
                            SMLoc &resultLoc) -> LogicalResult {
    auto [attr, attrLoc] = seqGroup.getAttribute(field);
    if (!attr) {
      if (required)
        return emitError(seqGroup.loc, "sequential cell missing '")
               << field << "'";
      return success();
    }
    auto str = dyn_cast<StringAttr>(attr);
    if (!str)
      return emitError(attrLoc, "sequential cell '")
             << field << "' must be a string";
    result = str;
    resultLoc = attrLoc;
    return success();
  };

  if (seqGroup.args.size() != 2 || !isa<StringAttr>(seqGroup.args[0]) ||
      !isa<StringAttr>(seqGroup.args[1]))
    return emitError(seqGroup.loc, "sequential group requires two string "
                                   "(var1, var2) arguments");
  StringRef var1 = cast<StringAttr>(seqGroup.args[0]).getValue();
  StringRef var2 = cast<StringAttr>(seqGroup.args[1]).getValue();

  // Optional asynchronous controls shared by `ff` and `latch`. Simultaneous
  // assertion of both is undefined in the state-element ops, matching the
  // assumption that sane designs never co-assert them.
  StringAttr clearAttr, presetAttr;
  SMLoc clearLoc, presetLoc;
  if (failed(
          getStringField("clear", /*required=*/false, clearAttr, clearLoc)) ||
      failed(
          getStringField("preset", /*required=*/false, presetAttr, presetLoc)))
    return failure();
  Value clearVal, presetVal;
  auto bindAsync = [&](StringAttr attr, SMLoc attrLoc, Value &out) {
    if (!attr) {
      out = Value();
      return success();
    }
    auto ref = resolvePinRef(attr.getValue(), attrLoc, portValues);
    if (failed(ref))
      return failure();
    out = activeHigh(ref->first, ref->second, loc);
    return success();
  };
  if (failed(bindAsync(clearAttr, clearLoc, clearVal)) ||
      failed(bindAsync(presetAttr, presetLoc, presetVal)))
    return failure();

  Value stateQ, stateQN;
  if (!isLatch) {
    StringAttr clockedOn, nextState;
    SMLoc clockedOnLoc, nextStateLoc;
    if (failed(getStringField("clocked_on", /*required=*/true, clockedOn,
                              clockedOnLoc)) ||
        failed(getStringField("next_state", /*required=*/true, nextState,
                              nextStateLoc)))
      return failure();
    if (seqGroup.getAttribute("clocked_on_also").first)
      return emitError(seqGroup.loc,
                       "unsupported 'clocked_on_also' in sequential cell");

    auto clockRef =
        resolvePinRef(clockedOn.getValue(), clockedOnLoc, portValues);
    if (failed(clockRef))
      return failure();
    auto edge =
        clockRef->second ? synth::EdgeKind::Fall : synth::EdgeKind::Rise;

    // Decompose `next_state` into data/enable/scan roles by matching mux
    // topologies. With Rd the data reference, Re the enable, Rsi/Rse the
    // scan pair (each a pin, optionally inverted with `!`), and IQ the
    // state variable, the supported shapes are:
    //   Rd | (Rd&!Rse)|(Rsi&Rse) | (Rd&!Rse)|(Rd&!Rsi)|(Rse&!Rsi) |
    //   (Rd&Re)|(IQ&!Re) | (Rd&Re&!Rse)|(IQ&!Re&!Rse)|(Rsi&Rse)
    // The middle scan form is the same mux written as a sum of products
    // (used by ASAP7); operators are normalized (`*`->`&`, `+`->`|`).
    // The scan mux (if any) is feedback-free and becomes a `comb.mux`; the
    // enable feedback becomes the `enable` operand, like `seq.compreg.ce`;
    // remaining inversions become `comb` negations. New topologies are a
    // hard error so that unseen Liberty shapes surface.
    std::string stripped;
    for (char c : nextState.getValue()) {
      if (isspace(c))
        continue;
      if (c == '*')
        stripped.push_back('&');
      else if (c == '+')
        stripped.push_back('|');
      else
        stripped.push_back(c);
    }

    // Scan pair discovery via `nextstate_type` pin attributes.
    StringRef scanIn, scanEn;
    for (const auto *pg : pinGroups) {
      auto [typeAttr, typeLoc] = pg->getAttribute("nextstate_type");
      if (!typeAttr)
        continue;
      auto typeStr = dyn_cast<StringAttr>(typeAttr);
      if (!typeStr)
        return emitError(typeLoc, "nextstate_type must be a string");
      StringRef pinName = cast<StringAttr>(pg->args[0]).getValue();
      if (typeStr.getValue() == "scan_in") {
        if (!scanIn.empty())
          return emitError(typeLoc, "multiple scan_in pins");
        scanIn = pinName;
      } else if (typeStr.getValue() == "scan_enable") {
        if (!scanEn.empty())
          return emitError(typeLoc, "multiple scan_enable pins");
        scanEn = pinName;
      }
    }
    if (scanIn.empty() != scanEn.empty())
      return emitError(seqGroup.loc, "scan_in without scan_enable (or vice "
                                     "versa)");

    // Candidate pins referenced by the expression, minus the state vars.
    // Every remaining reference must be an input pin.
    SmallVector<StringRef> refs;
    StringRef rest(stripped);
    while (!rest.empty()) {
      size_t len = 0;
      while (len < rest.size() &&
             (isalnum(rest[len]) || rest[len] == '_' || rest[len] == '[' ||
              rest[len] == ']' || rest[len] == '.'))
        ++len;
      if (len == 0) {
        rest = rest.drop_front();
        continue;
      }
      StringRef tok = rest.take_front(len);
      rest = rest.drop_front(len);
      if (tok != var1 && tok != var2 && !llvm::is_contained(refs, tok))
        refs.push_back(tok);
    }
    for (StringRef ref : refs)
      if (!portValues.count(ref))
        return emitError(nextStateLoc, "unsupported next_state expression '")
               << nextState.getValue() << "': unknown pin '" << ref << "'";

    Value dataVal, enableVal;
    bool matched = false;
    auto setMatch = [&](Value data, Value enable) -> LogicalResult {
      if (matched)
        return emitError(nextStateLoc, "ambiguous next_state expression '")
               << nextState.getValue() << "'";
      matched = true;
      dataVal = data;
      enableVal = enable;
      return success();
    };
    // Scan pair candidates: the `nextstate_type` pins when present (not all
    // scan cells carry them), otherwise every ordered pin pair. Trying the
    // no-scan option covers cells whose scan inputs exist but are unused by
    // `next_state`.
    SmallVector<std::pair<StringRef, StringRef>> scanOptions;
    scanOptions.emplace_back(StringRef(), StringRef());
    if (!scanIn.empty()) {
      scanOptions.emplace_back(scanIn, scanEn);
    } else {
      for (StringRef a : refs)
        for (StringRef b : refs)
          if (a != b)
            scanOptions.emplace_back(a, b);
    }
    auto buildScanMux = [&](Value seVal, Value siVal, Value base) {
      return comb::MuxOp::create(builder, loc, seVal, siVal, base);
    };
    // A role reference with polarity, e.g. (D,false) or (SCE,true) for !SCE.
    auto refStr = [&](StringRef pin, bool inv) -> std::string {
      return (inv ? "!" : "") + pin.str();
    };
    auto negate = [&](const std::string &r) -> std::string {
      if (!r.empty() && r[0] == '!')
        return r.substr(1);
      return "!" + r;
    };
    for (StringRef dataCand : refs) {
      if (dataCand == scanIn || dataCand == scanEn)
        continue;
      // Candidate enable pins: remaining referenced pins.
      SmallVector<StringRef> enOptions;
      enOptions.push_back(StringRef());
      for (StringRef enCand : refs)
        if (enCand != dataCand && enCand != scanIn && enCand != scanEn)
          enOptions.push_back(enCand);
      for (StringRef enCand : enOptions) {
        for (auto [siCand, seCand] : scanOptions) {
          if (!siCand.empty() && (siCand == dataCand || siCand == enCand ||
                                  seCand == dataCand || seCand == enCand))
            continue;
          // Polarity trials for every role.
          for (bool dInv : {false, true}) {
            for (bool eInv : {false, true}) {
              if (enCand.empty() && eInv)
                continue;
              for (bool siInv : {false, true}) {
                for (bool seInv : {false, true}) {
                  if (siCand.empty() && (siInv || seInv))
                    continue;
                  std::string rd = refStr(dataCand, dInv);
                  std::string re = refStr(enCand, eInv);
                  std::string rsi = refStr(siCand, siInv);
                  std::string rse = refStr(seCand, seInv);
                  // Each candidate pairs an expected string with whether
                  // the scan value is negated relative to its reference.
                  // That is the case only for the sum-of-products scan
                  // form: (Rd&!Rse)|(Rd&!Rsi)|(Rse&!Rsi) is mux(Rse, !Rsi,
                  // Rd) (the third term is the consensus term).
                  SmallVector<std::pair<std::string, bool>> expected;
                  if (enCand.empty() && siCand.empty()) {
                    expected.emplace_back(rd, false);
                  } else if (enCand.empty()) {
                    expected.emplace_back("(" + rd + "&" + negate(rse) + ")|(" +
                                              rsi + "&" + rse + ")",
                                          false);
                    expected.emplace_back("(" + rd + "&" + negate(rse) + ")|(" +
                                              rd + "&" + negate(rsi) + ")|(" +
                                              rse + "&" + negate(rsi) + ")",
                                          true);
                  } else if (siCand.empty()) {
                    expected.emplace_back("(" + rd + "&" + re + ")|(" +
                                              var1.str() + "&" + negate(re) +
                                              ")",
                                          false);
                  } else {
                    expected.emplace_back(
                        "(" + rd + "&" + re + "&" + negate(rse) + ")|(" +
                            var1.str() + "&" + negate(re) + "&" + negate(rse) +
                            ")|(" + rsi + "&" + rse + ")",
                        false);
                  }
                  for (const auto &[candidate, flipSI] : expected) {
                    if (stripped != candidate)
                      continue;
                    Value dataOp =
                        activeHigh(portValues.lookup(dataCand), dInv, loc);
                    if (!siCand.empty())
                      dataOp = buildScanMux(
                          activeHigh(portValues.lookup(seCand), seInv, loc),
                          activeHigh(portValues.lookup(siCand), siInv != flipSI,
                                     loc),
                          dataOp);
                    Value enOp =
                        enCand.empty()
                            ? Value()
                            : activeHigh(portValues.lookup(enCand), eInv, loc);
                    if (failed(setMatch(dataOp, enOp)))
                      return failure();
                  }
                }
              }
            }
          }
        }
      }
    }
    if (!matched)
      return emitError(nextStateLoc, "unsupported next_state expression '")
             << nextState.getValue() << "'";

    auto edgeAttr = synth::EdgeKindAttr::get(builder.getContext(), edge);
    auto ffOp = synth::FFOp::create(builder, loc, dataVal, clockRef->first,
                                    enableVal, clearVal, presetVal, edgeAttr);
    stateQ = ffOp.getQ();
    stateQN = ffOp.getQn();
  } else {
    StringAttr enableAttr, dataInAttr;
    SMLoc enableLoc, dataInLoc;
    if (failed(getStringField("enable", /*required=*/true, enableAttr,
                              enableLoc)) ||
        failed(getStringField("data_in", /*required=*/true, dataInAttr,
                              dataInLoc)))
      return failure();
    auto enRef = resolvePinRef(enableAttr.getValue(), enableLoc, portValues);
    auto dataRef = resolvePinRef(dataInAttr.getValue(), dataInLoc, portValues);
    if (failed(enRef) || failed(dataRef))
      return failure();
    auto latchOp = synth::LatchOp::create(
        builder, loc, activeHigh(dataRef->first, dataRef->second, loc),
        activeHigh(enRef->first, enRef->second, loc), clearVal, presetVal);
    stateQ = latchOp.getQ();
    stateQN = latchOp.getQn();
  }

  // Output pins select the state variable recorded in `function`.
  SmallVector<Value> outputs;
  for (const auto &port : ports) {
    if (port.dir != hw::ModulePort::Direction::Output)
      continue;
    auto *it = llvm::find_if(pinGroups, [&](const LibertyGroup *g) {
      assert(g->name == "pin" && "expected pin group");
      return cast<StringAttr>(g->args[0]) == port.name;
    });
    if (!it)
      continue;
    const LibertyGroup *pg = *it;
    auto [funcAttr, funcLoc] = pg->getAttribute("function");
    assert(funcAttr && "output pin missing function attribute");
    auto funcStr = dyn_cast<StringAttr>(funcAttr);
    if (!funcStr)
      return emitError(funcLoc, "sequential output function must be a string");
    if (funcStr.getValue() == var1)
      outputs.push_back(stateQ);
    else if (funcStr.getValue() == var2)
      outputs.push_back(stateQN);
    else
      return emitError(funcLoc, "unsupported sequential output function '")
             << funcStr.getValue() << "': expected '" << var1 << "' or '"
             << var2 << "'";
  }

  moduleOp.getBodyBlock()->getTerminator()->setOperands(outputs);
  return success();
}

LogicalResult
LibertyParser::buildCellLogic(hw::HWModuleOp moduleOp,
                              ArrayRef<hw::PortInfo> ports,
                              ArrayRef<const LibertyGroup *> pinGroups) {

  // All outputs have "function": emit logic into hw.module body.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());

  llvm::StringMap<Value> portValues;
  for (const auto &port : ports)
    if (port.dir == hw::ModulePort::Direction::Input)
      portValues[port.name.getValue()] =
          moduleOp.getBodyBlock()->getArgument(port.argNum);

  SmallVector<Value> outputs;
  for (const auto &port : ports) {
    if (port.dir != hw::ModulePort::Direction::Output)
      continue;

    auto *it = llvm::find_if(pinGroups, [&](const LibertyGroup *g) {
      assert(g->name == "pin" && "expected pin group");
      // First arg is the pin name
      return cast<StringAttr>(g->args[0]) == port.name;
    });

    if (!it)
      continue;

    const LibertyGroup *pg = *it;
    auto attrPair = pg->getAttribute("function");
    assert(attrPair.first && "output pin missing function attribute");

    Value val;
    if (parseExpression(val, cast<StringAttr>(attrPair.first).getValue(),
                        portValues, attrPair.second))
      return failure();
    outputs.push_back(val);
  }

  moduleOp.getBodyBlock()->getTerminator()->setOperands(outputs);
  return success();
}

LogicalResult LibertyParser::buildClockGateCellLogic(
    hw::HWModuleOp moduleOp, ArrayRef<hw::PortInfo> ports,
    ArrayRef<const LibertyGroup *> pinGroups, const LibertyGroup &cellGroup) {
  // An integrated clock gate is a transparent latch plus an AND gate. Only
  // the latch itself is a new op (`synth.latch`); the AND, the test muxing,
  // and any inversions are plain `comb` operations.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());
  auto loc = lexer.translateLocation(cellGroup.loc);

  llvm::StringMap<Value> portValues;
  for (const auto &port : ports)
    if (port.dir == hw::ModulePort::Direction::Input)
      portValues[port.name.getValue()] =
          moduleOp.getBodyBlock()->getArgument(port.argNum);

  auto isSet = [&](const LibertyGroup &pinGroup, StringRef attr) {
    auto str = dyn_cast_or_null<StringAttr>(pinGroup.getAttribute(attr).first);
    return str && str.getValue() == "true";
  };

  // Discover roles from clock_gate_* pin attributes. The internal state node
  // is not a port; it is identified by name only.
  StringRef clockPin, enablePin, testPin, outPin, nodeName;
  const LibertyGroup *outPinGroup = nullptr;
  for (const auto &sub : cellGroup.subGroups) {
    if (sub->name != "pin" || sub->args.empty())
      continue;
    auto pinName = dyn_cast<StringAttr>(sub->args[0]);
    if (!pinName)
      return emitError(sub->loc, "pin missing name");
    auto takeRole = [&](StringRef attr, StringRef &role) -> LogicalResult {
      if (!isSet(*sub, attr))
        return success();
      if (!role.empty())
        return emitError(sub->loc, "multiple pins with '") << attr << "'";
      role = pinName.getValue();
      return success();
    };
    if (failed(takeRole("clock_gate_clock_pin", clockPin)) ||
        failed(takeRole("clock_gate_enable_pin", enablePin)) ||
        failed(takeRole("clock_gate_test_pin", testPin)) ||
        failed(takeRole("clock_gate_out_pin", outPin)))
      return failure();
    if (pinName.getValue() == outPin)
      outPinGroup = sub.get();
    auto [dirAttr, _dirLoc] = sub->getAttribute("direction");
    auto dirStr = dyn_cast_or_null<StringAttr>(dirAttr);
    if (dirStr && dirStr.getValue() == "internal") {
      auto [nodeAttr, nodeLoc] = sub->getAttribute("internal_node");
      auto nodeStr = dyn_cast_or_null<StringAttr>(nodeAttr);
      if (!nodeStr)
        return emitError(sub->loc, "internal pin misses 'internal_node'");
      if (!nodeName.empty() && nodeName != nodeStr.getValue())
        return emitError(sub->loc, "multiple internal nodes");
      nodeName = nodeStr.getValue();
    }
  }
  if (clockPin.empty() || enablePin.empty() || outPin.empty())
    return emitError(cellGroup.loc, "integrated clock gate misses clock, "
                                    "enable, or output pin roles");
  if (!portValues.count(clockPin) || !portValues.count(enablePin) ||
      (!testPin.empty() && !portValues.count(testPin)))
    return emitError(cellGroup.loc, "clock-gate control pin is not a module "
                                    "input");
  size_t numOutput = llvm::count_if(
      ports, [](const hw::PortInfo &port) { return port.isOutput(); });
  if (numOutput != 1 || !outPinGroup)
    return emitError(cellGroup.loc,
                     "integrated clock gate requires a single output pin");
  Value clkVal = portValues.lookup(clockPin);

  // Latch data and transparency, from the statetable or from the
  // clock_gating_integrated_cell architecture.
  const LibertyGroup *tableGroup = nullptr;
  for (const auto &sub : cellGroup.subGroups)
    if (sub->name == "statetable") {
      if (tableGroup)
        return emitError(sub->loc, "multiple statetable groups");
      tableGroup = sub.get();
    }
  auto [archAttr, archLoc] =
      cellGroup.getAttribute("clock_gating_integrated_cell");

  synth::ClockGateOp gateOp;
  if (tableGroup) {
    // When both descriptions are present, the table is authoritative and
    // the architecture cross-checks it. Observed values are all
    // latch-posedge family (quoted or bare).
    if (archAttr) {
      auto archStr = dyn_cast_or_null<StringAttr>(archAttr);
      if (!archStr || (archStr.getValue() != "latch_posedge" &&
                       archStr.getValue() != "latch_posedge_precontrol"))
        return emitError(archLoc, "unsupported clock-gating architecture");
    }
    // Header: ("<inputs...>", "<statevar>") with space-separated inputs.
    if (tableGroup->args.size() != 2 || !isa<StringAttr>(tableGroup->args[0]) ||
        !isa<StringAttr>(tableGroup->args[1]))
      return emitError(tableGroup->loc, "statetable requires (inputs, "
                                        "statevar) string arguments");
    SmallVector<StringRef> tableInputs;
    cast<StringAttr>(tableGroup->args[0])
        .getValue()
        .split(tableInputs, ' ', -1, false);
    StringRef stateVar = cast<StringAttr>(tableGroup->args[1]).getValue();
    if (tableInputs.empty())
      return emitError(tableGroup->loc, "statetable misses input pins");
    if (!nodeName.empty() && nodeName != stateVar)
      return emitError(tableGroup->loc, "internal node disagrees with "
                                        "statetable variable");
    nodeName = stateVar;
    auto [tableAttr, tableLoc] = tableGroup->getAttribute("table");
    auto tableStr = dyn_cast_or_null<StringAttr>(tableAttr);
    if (!tableStr)
      return emitError(tableGroup->loc, "statetable misses 'table' string");

    // Rows look like `L H : - : H`. Only steady H/L/- inputs and H/L/N
    // next states are supported; anything else is not a simple latch.
    struct TableRow {
      std::string inputs;
      char next;
    };
    SmallVector<TableRow> rows;
    std::string text = tableStr.getValue().str();
    std::string flat;
    for (size_t i = 0; i < text.size();) {
      if (text[i] == '\\' && i + 1 < text.size() && text[i + 1] == '\n') {
        i += 2;
        continue;
      }
      flat.push_back(text[i++]);
    }
    SmallVector<StringRef> chunks;
    StringRef(flat).split(chunks, ',', -1, false);
    for (StringRef chunk : chunks) {
      SmallVector<StringRef> toks;
      chunk.split(toks, ' ', -1, false);
      if (toks.size() != tableInputs.size() + 4 ||
          toks[tableInputs.size()] != ":" ||
          toks[tableInputs.size() + 1] != "-" ||
          toks[tableInputs.size() + 2] != ":")
        return emitError(tableLoc, "unsupported statetable row '")
               << chunk.trim() << "'";
      std::string inputs;
      for (size_t k = 0; k < tableInputs.size(); ++k) {
        if (toks[k] != "H" && toks[k] != "L" && toks[k] != "-" &&
            toks[k] != "?")
          return emitError(tableLoc, "unsupported statetable input '")
                 << toks[k] << "'";
        inputs.push_back(toks[k][0]);
      }
      char next = toks.back()[0];
      if ((toks.back().size() != 1) ||
          (next != 'H' && next != 'L' && next != 'N'))
        return emitError(tableLoc, "unsupported statetable next state '")
               << toks.back() << "'";
      rows.push_back({inputs, next});
    }

    // Expand don't-cares; Liberty evaluates rows top-down, first match wins.
    llvm::StringMap<char> nextOf;
    for (const auto &row : rows) {
      SmallVector<std::string> assigns = {""};
      for (char v : row.inputs) {
        if (v == 'H' || v == 'L') {
          for (auto &a : assigns)
            a.push_back(v == 'H' ? '1' : '0');
        } else {
          SmallVector<std::string> grown;
          for (auto &a : assigns) {
            grown.push_back(a + "0");
            grown.push_back(a + "1");
          }
          assigns = std::move(grown);
        }
        if (assigns.size() > 1024)
          return emitError(tableLoc, "statetable too large to expand");
      }
      for (auto &a : assigns)
        if (!nextOf.count(a))
          nextOf[a] = row.next;
    }

    // The clock is the unique input held (N) at one level and assigned at
    // the other; the latch is transparent at the assigned level.
    StringRef latchClock;
    char transparentLevel = 0;
    for (size_t c = 0; c < tableInputs.size(); ++c) {
      for (char t : {'L', 'H'}) {
        bool ok = true, hasTransparent = false;
        for (const auto &[assign, next] : nextOf) {
          bool atLevel = assign[c] == (t == 'H' ? '1' : '0');
          if (atLevel) {
            if (next == 'N') {
              ok = false;
              break;
            }
            hasTransparent = true;
          } else if (next != 'N') {
            ok = false;
            break;
          }
        }
        if (ok && hasTransparent) {
          if (!latchClock.empty())
            return emitError(tableLoc, "statetable is not a simple latch");
          latchClock = tableInputs[c];
          transparentLevel = t;
        }
      }
    }
    if (latchClock.empty())
      return emitError(tableLoc, "statetable is not a simple latch");
    if (latchClock != clockPin)
      return emitError(tableLoc, "statetable clock disagrees with clock-gate "
                                 "clock pin");
    if (archAttr && transparentLevel != 'L')
      return emitError(tableLoc, "statetable disagrees with "
                                 "latch_posedge_precontrol architecture");

    // Data must be OR(enable[, test]) over exactly those pins; anything
    // else is not a clock gate this op can express.
    llvm::DenseSet<StringRef> dataSet;
    for (size_t k = 0; k < tableInputs.size(); ++k)
      if (tableInputs[k] != latchClock) {
        if (!portValues.count(tableInputs[k]))
          return emitError(tableLoc, "statetable references unknown pin '")
                 << tableInputs[k] << "'";
        dataSet.insert(tableInputs[k]);
      }
    size_t wantRoles = testPin.empty() ? 1 : 2;
    if (dataSet.size() != wantRoles || !dataSet.count(enablePin) ||
        (!testPin.empty() && !dataSet.count(testPin)))
      return emitError(tableLoc, "clock-gate data is not OR(enable[, test])");
    size_t clockPos = tableInputs.size(), enPos = tableInputs.size(),
           tePos = tableInputs.size();
    for (size_t k = 0; k < tableInputs.size(); ++k) {
      if (tableInputs[k] == latchClock)
        clockPos = k;
      if (tableInputs[k] == enablePin)
        enPos = k;
      if (!testPin.empty() && tableInputs[k] == testPin)
        tePos = k;
    }
    char transparentBit = (transparentLevel == 'H' ? '1' : '0');
    for (const auto &[assign, next] : nextOf) {
      if (assign[clockPos] != transparentBit)
        continue; // hold rows (established by the detector)
      bool wantHigh =
          assign[enPos] == '1' || (!testPin.empty() && assign[tePos] == '1');
      if ((next == 'H') != wantHigh)
        return emitError(tableLoc, "clock-gate data is not OR(enable[, test])");
    }
    auto tableLevel = transparentLevel == 'L' ? synth::ClockGateLevel::Low
                                              : synth::ClockGateLevel::High;
    gateOp = synth::ClockGateOp::create(
        builder, loc, clkVal, portValues.lookup(enablePin),
        testPin.empty() ? Value() : portValues.lookup(testPin),
        synth::ClockGateLevelAttr::get(builder.getContext(), tableLevel));
  } else {
    // Without a table, only precontrol (test ORed into latch data,
    // transparent while low) is supported.
    auto archStr = dyn_cast_or_null<StringAttr>(archAttr);
    if (!archStr || archStr.getValue() != "latch_posedge_precontrol")
      return emitError(archLoc, "unsupported clock-gating architecture");
    gateOp = synth::ClockGateOp::create(
        builder, loc, clkVal, portValues.lookup(enablePin),
        testPin.empty() ? Value() : portValues.lookup(testPin),
        synth::ClockGateLevelAttr::get(builder.getContext(),
                                       synth::ClockGateLevel::Low));
  }
  if (nodeName.empty())
    return emitError(cellGroup.loc, "cannot identify clock-gate state node");

  // Output: the op result. An explicit function/state_function must be
  // the plain AND(clock, state); anything else contradicts the op semantics.
  auto [funcAttr, funcLoc] = outPinGroup->getAttribute("function");
  if (!funcAttr)
    std::tie(funcAttr, funcLoc) = outPinGroup->getAttribute("state_function");
  if (funcAttr) {
    auto funcStr = dyn_cast<StringAttr>(funcAttr);
    if (!funcStr)
      return emitError(funcLoc, "clock-gate output function must be a string");
    std::string stripped;
    for (char c : funcStr.getValue()) {
      if (isspace(c))
        continue;
      stripped.push_back(c == '*' ? '&' : c);
    }
    // A single pair of redundant parentheses, as in "(CLK*M0)".
    while (stripped.size() >= 2 && stripped.front() == '(' &&
           stripped.back() == ')') {
      int depth = 0;
      bool balanced = true;
      for (size_t i = 0; i + 1 < stripped.size(); ++i) {
        if (stripped[i] == '(')
          ++depth;
        else if (stripped[i] == ')') {
          if (--depth == 0) {
            balanced = false;
            break;
          }
        }
      }
      if (!balanced)
        break;
      stripped = stripped.substr(1, stripped.size() - 2);
    }
    SmallVector<StringRef> parts;
    StringRef(stripped).split(parts, '&', -1, false);
    llvm::DenseSet<StringRef> operands(parts.begin(), parts.end());
    if (parts.size() != 2 || operands.size() != 2 ||
        !operands.count(clockPin) || !operands.count(nodeName))
      return emitError(funcLoc, "unsupported clock-gate output function '")
             << funcStr.getValue() << "': expected AND(clock, state)";
  }

  moduleOp.getBodyBlock()->getTerminator()->setOperands({gateOp.getGclk()});
  return success();
}

ParseResult LibertyParser::lowerCell(
    const LibertyGroup &group, StringAttr &cellNameAttr,
    const llvm::StringMap<const LibertyGroup *> &tableTemplates) {

  if (group.args.empty())
    return emitError(group.loc, "cell missing name");
  cellNameAttr = dyn_cast<StringAttr>(group.args[0]);
  if (!cellNameAttr)
    return emitError(group.loc, "cell name must be a string");

  SmallVector<hw::PortInfo> ports;
  SmallVector<const LibertyGroup *> pinGroups;
  SmallVector<Attribute> timingArcs;
  size_t numOutputPinMissingFunction = 0;
  bool hasThreeStateOutput = false;

  if (failed(collectPorts(group, tableTemplates, ports, pinGroups, timingArcs,
                          numOutputPinMissingFunction, hasThreeStateOutput)))
    return failure();

  // Fix up argNum for inputs
  int inputIdx = 0;
  for (auto &p : ports)
    if (p.dir == hw::ModulePort::Direction::Input)
      p.argNum = inputIdx++;
    else
      p.argNum = 0;

  auto loc = lexer.translateLocation(group.loc);
  size_t numOutput =
      static_cast<size_t>(llvm::count_if(ports, [](const hw::PortInfo &port) {
        return port.dir == hw::ModulePort::Direction::Output;
      }));

  // A cell with a single `ff` or `latch` group models its state variables
  // (for example, a Sky130 flop uses `IQ` and `IQ_N`) with a
  // `synth.ff`/`synth.latch` state element in an `hw.module`
  // body. Combinational parts (scan muxes, active-low inversions, output
  // selection) are plain `comb`/`hw` operations; only the state element
  // itself is a new op. Cells with `statetable` groups that do not decode
  // to a clock gate, multiple state groups, `inout` ports, or missing
  // output functions stay black boxes.
  const LibertyGroup *ffGroup = nullptr;
  const LibertyGroup *latchGroup = nullptr;
  int numFF = 0, numLatch = 0;
  bool hasStatetable = false;
  for (const auto &sub : group.subGroups) {
    if (sub->name == "ff") {
      ffGroup = sub.get();
      ++numFF;
    } else if (sub->name == "latch") {
      latchGroup = sub.get();
      ++numLatch;
    } else if (sub->name == "statetable") {
      hasStatetable = true;
    }
  }
  bool hasSequentialModel = numFF + numLatch > 0 || hasStatetable;
  bool hasInOutPort = llvm::any_of(ports, [](const hw::PortInfo &port) {
    return isa<hw::InOutType>(port.type);
  });
  // Integrated clock gates are claimed by pin/cell attributes (not by
  // output functions, which a clock-gate output may not even have). The
  // body builder validates the architecture strictly.
  bool hasClockGate = false;
  for (const auto &sub : group.subGroups) {
    if (sub->name != "pin" || sub->args.empty())
      continue;
    auto [outAttr, _] = sub->getAttribute("clock_gate_out_pin");
    auto outStr = dyn_cast_or_null<StringAttr>(outAttr);
    if (outStr && outStr.getValue() == "true")
      hasClockGate = true;
  }
  if (static_cast<bool>(
          group.getAttribute("clock_gating_integrated_cell").first))
    hasClockGate = true;

  // Cell-level scalar attributes the importer does not otherwise consume
  // (area, dont_use, clock-gating architecture, ...) are preserved as a
  // generic dictionary for inspection and future cost models. Behavioral
  // lowering consumes bodies and timing arcs instead.
  DictionaryAttr cellDict;
  if (!group.attrs.empty()) {
    SmallVector<NamedAttribute> cellAttrList;
    for (const auto &attr : group.attrs)
      cellAttrList.push_back(builder.getNamedAttr(attr.name, attr.value));
    cellDict = builder.getDictionaryAttr(cellAttrList);
  }
  auto attachCellDict = [&](Operation *op) {
    if (cellDict)
      op->setAttr("synth.liberty.cell", cellDict);
  };

  // Decide whether to create hw.module (with logic) or hw.module.extern (black
  // box) based on whether output pins have "function" attributes.
  // All outputs must either have "function" or all must lack it.

  // Mixed case: some outputs have "function", others don't - error.
  if (!hasSequentialModel && !hasInOutPort &&
      numOutputPinMissingFunction != 0 &&
      numOutputPinMissingFunction != numOutput)
    return emitError(group.loc, "cell '")
           << cellNameAttr.getValue()
           << "' has mixed output pins with and without 'function' attribute";

  // Integrated clock gates get a latch-plus-AND body. Anything the body
  // builder rejects is a hard error, so unseen clock-gate architectures
  // surface instead of silently degrading.
  if (hasClockGate && !hasThreeStateOutput) {
    auto moduleOp = hw::HWModuleOp::create(builder, loc, cellNameAttr, ports);
    if (failed(buildClockGateCellLogic(moduleOp, ports, pinGroups, group)))
      return failure();
    if (!timingArcs.empty())
      moduleOp->setAttr("synth.timing.nldm", builder.getArrayAttr(timingArcs));
    attachCellDict(moduleOp);
    return success();
  }

  // Single `ff`/`latch` cells with fully-specified output functions get an
  // executable body. Anything the body builder rejects (for example an
  // exotic `next_state` shape) is a hard error, so new Liberty shapes
  // surface instead of silently degrading.
  if (!hasInOutPort && !hasStatetable && !hasClockGate &&
      !hasThreeStateOutput && numFF + numLatch == 1 &&
      numOutputPinMissingFunction == 0 && numOutput > 0) {
    auto moduleOp = hw::HWModuleOp::create(builder, loc, cellNameAttr, ports);
    const LibertyGroup &seqGroup = numFF ? *ffGroup : *latchGroup;
    if (failed(buildSequentialCellLogic(moduleOp, ports, pinGroups, seqGroup,
                                        /*isLatch=*/numLatch == 1)))
      return failure();
    if (!timingArcs.empty())
      moduleOp->setAttr("synth.timing.nldm", builder.getArrayAttr(timingArcs));
    attachCellDict(moduleOp);
    return success();
  }

  // All outputs lack "function", the cell is sequential but not covered by
  // the body builders above, or an output is three-state: emit as
  // hw.module.extern (black box).
  if (hasSequentialModel || hasThreeStateOutput ||
      (hasInOutPort && numOutputPinMissingFunction != 0) ||
      numOutputPinMissingFunction == numOutput) {
    auto externOp =
        hw::HWModuleExternOp::create(builder, loc, cellNameAttr, ports);
    if (!timingArcs.empty())
      externOp->setAttr("synth.timing.nldm", builder.getArrayAttr(timingArcs));
    attachCellDict(externOp);
    // Preserve the sequential model (`ff`, `latch`, `statetable`) groups so
    // that a sequential mapping pass can match register/latch cells.  The
    // groups are stored keyed by group name, following the convention used
    // for the generic Liberty attributes.
    llvm::StringMap<SmallVector<Attribute>> seqGroups;
    for (const auto &sub : group.subGroups)
      if (sub->name == "ff" || sub->name == "latch" ||
          sub->name == "statetable")
        seqGroups[sub->name].push_back(convertGroupToAttr(*sub));
    if (!seqGroups.empty()) {
      SmallVector<NamedAttribute> seqAttrs;
      for (auto &it : seqGroups)
        seqAttrs.push_back(builder.getNamedAttr(
            it.getKey(), builder.getArrayAttr(it.getValue())));
      externOp->setAttr("synth.liberty.sequential",
                        builder.getDictionaryAttr(seqAttrs));
    }
    return success();
  }

  // All outputs have "function": emit as hw.module with logic.
  auto moduleOp = hw::HWModuleOp::create(builder, loc, cellNameAttr, ports);

  if (failed(buildCellLogic(moduleOp, ports, pinGroups)))
    return failure();

  if (!timingArcs.empty())
    moduleOp->setAttr("synth.timing.nldm", builder.getArrayAttr(timingArcs));
  attachCellDict(moduleOp);

  return success();
}

ParseResult LibertyParser::parseGroupBody(LibertyGroup &group) {
  // Parse args: ( arg1, arg2 )
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    lexer.nextToken(); // (
    while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
      Attribute arg;
      if (parseAttribute(arg))
        return failure();
      group.args.push_back(arg);
      if (lexer.peekToken().kind == LibertyTokenKind::Comma)
        lexer.nextToken();
    }
    if (consume(LibertyTokenKind::RParen, "expected ')'"))
      return failure();
  }

  // Parse body: { ... }
  if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
    lexer.nextToken(); // {
    while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
      if (parseStatement(group))
        return failure();
    }
    if (consume(LibertyTokenKind::RBrace, "expected '}'"))
      return failure();
  } else {
    // Optional semicolon if no body
    if (lexer.peekToken().kind == LibertyTokenKind::Semi)
      lexer.nextToken();
  }
  return success();
}

// Parse group, attribute, or define statements
ParseResult LibertyParser::parseStatement(LibertyGroup &parent) {
  auto nameTok = lexer.nextToken();
  if (nameTok.kind != LibertyTokenKind::Identifier)
    return emitError(nameTok.location, "expected identifier");
  StringRef name = nameTok.spelling;

  if (name == "define" && lexer.peekToken().kind != LibertyTokenKind::LParen)
    return emitError(nameTok.location, "expected '(' after `define`");

  // Attribute statement.
  if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
    lexer.nextToken(); // :
    Attribute val;
    auto loc = lexer.peekToken().location;
    if (parseAttribute(val))
      return failure();
    parent.attrs.emplace_back(name, val, loc);
    if (lexer.peekToken().kind == LibertyTokenKind::Semi) {
      lexer.nextToken();
      return success();
    }
    // Shipped PDKs systematically omit the ';' after some attributes
    // (ASAP7 r28 cell `area`). After a complete attribute value only ';'
    // may validly follow, so an identifier (next statement) or '}' (end of
    // group) unambiguously means the terminator was omitted. Recover with
    // a warning; anything else stays a hard error.
    if (lexer.peekToken().kind == LibertyTokenKind::Identifier ||
        lexer.peekToken().kind == LibertyTokenKind::RBrace) {
      emitWarning(lexer.peekToken().location, "missing ';' after attribute '")
          << name << "'; assuming terminator";
      return success();
    }
    return consume(LibertyTokenKind::Semi, "expected ';'");
  }

  // Group statement.
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    // `define` declarations are semicolon-terminated groups rather than
    // ordinary attributes.  They are not needed to lower cells, but keeping
    // them as generic groups lets clients inspect the declarations after
    // import.
    auto subGroup = std::make_unique<LibertyGroup>();
    subGroup->name = name;
    subGroup->loc = nameTok.location;
    if (parseGroupBody(*subGroup))
      return failure();
    parent.subGroups.push_back(std::move(subGroup));
    return success();
  }

  return emitError(nameTok.location, "expected ':' or '('");
}

// Parse an attribute value, which can be:
// - A string: "value"
// - A number: 1.23
// - An identifier: value
ParseResult LibertyParser::parseAttribute(Attribute &result) {
  auto token = lexer.peekToken();
  // Number token
  if (token.is(LibertyTokenKind::Number)) {
    lexer.nextToken();
    StringRef numStr = token.spelling;

    // lexNumber also consumes an optional unit suffix.  Parse only the
    // numeric prefix here; Liberty's values are interpreted in the units
    // declared by the containing library.
    size_t numericEnd = 0;
    if (numericEnd < numStr.size() &&
        (numStr[numericEnd] == '+' || numStr[numericEnd] == '-'))
      ++numericEnd;
    while (numericEnd < numStr.size() && isdigit(numStr[numericEnd]))
      ++numericEnd;
    if (numericEnd < numStr.size() && numStr[numericEnd] == '.') {
      ++numericEnd;
      while (numericEnd < numStr.size() && isdigit(numStr[numericEnd]))
        ++numericEnd;
    }
    if (numericEnd < numStr.size() &&
        (numStr[numericEnd] == 'e' || numStr[numericEnd] == 'E')) {
      ++numericEnd;
      if (numericEnd < numStr.size() &&
          (numStr[numericEnd] == '+' || numStr[numericEnd] == '-'))
        ++numericEnd;
      while (numericEnd < numStr.size() && isdigit(numStr[numericEnd]))
        ++numericEnd;
    }

    StringRef numeric = numStr.take_front(numericEnd);
    double val;
    if (!numeric.getAsDouble(val)) {
      result = builder.getF64FloatAttr(val);
      return success();
    }
    return emitError(token.location, "expected number value");
  }

  // Identifier token
  if (token.is(LibertyTokenKind::Identifier)) {
    lexer.nextToken();
    result = builder.getStringAttr(token.spelling);
    return success();
  }

  if (token.is(LibertyTokenKind::String)) {
    lexer.nextToken();
    result = builder.getStringAttr(getTokenSpelling(token));
    return success();
  }

  return emitError(token.location, "expected attribute value");
}

// Parse boolean expressions from Liberty function attributes
// Supports: *, +, ^, !, (), and identifiers
ParseResult LibertyParser::parseExpression(Value &result, StringRef expr,
                                           const llvm::StringMap<Value> &values,
                                           SMLoc loc) {
  ExpressionParser parser(lexer, builder, expr, values, loc);
  return parser.parse(result);
}

} // namespace

namespace circt::liberty {
LogicalResult importLibertyFile(ModuleOp module, StringRef filename) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file)
    return emitError(UnknownLoc::get(module.getContext())) << errorMessage;
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  MLIRContext *context = module.getContext();
  context->loadDialect<hw::HWDialect>();
  context->loadDialect<comb::CombDialect>();
  context->loadDialect<synth::SynthDialect>();
  LibertyParser parser(sourceMgr, context, module);
  return parser.parse();
}

void registerImportLibertyTranslation() {
  TranslateToMLIRRegistration reg(
      "import-liberty", "Import Liberty file",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        ModuleOp module = ModuleOp::create(UnknownLoc::get(context));
        LibertyParser parser(sourceMgr, context, module);
        // Load required dialects
        context->loadDialect<hw::HWDialect>();
        context->loadDialect<comb::CombDialect>();
        context->loadDialect<synth::SynthDialect>();

        if (failed(parser.parse()))
          return OwningOpRef<ModuleOp>();
        return OwningOpRef<ModuleOp>(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<HWDialect>();
        registry.insert<comb::CombDialect>();
        registry.insert<synth::SynthDialect>();
      });
}
} // namespace circt::liberty
