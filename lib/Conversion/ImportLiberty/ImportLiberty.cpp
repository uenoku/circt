//===- ImportLiberty.cpp - Liberty to CIRCT Import ------------------------===//
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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
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
  LBrace,     // {
  RBrace,     // }
  LParen,     // (
  RParen,     // )
  Colon,      // :
  Semi,       // ;
  Comma,      // ,
  Plus,       // +
  Minus,      // -
  Star,       // *
  Slash,      // /
  Exclaim,    // !
  Apostrophe, // '

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
  case LibertyTokenKind::Plus:
    return "'+'";
  case LibertyTokenKind::Minus:
    return "'-'";
  case LibertyTokenKind::Star:
    return "'*'";
  case LibertyTokenKind::Slash:
    return "'/'";
  case LibertyTokenKind::Exclaim:
    return "'!'";
  case LibertyTokenKind::Apostrophe:
    return "'''";
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

  Location translateLocation(llvm::SMLoc loc) {
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
  LibertyToken makeToken(LibertyTokenKind kind, const char *start);
};

struct PinDef {
  StringRef name;
  bool isInput = false;
  bool isOutput = false;
  std::string function;
  SmallVector<NamedAttribute> attrs;
  llvm::StringMap<SmallVector<SmallVector<NamedAttribute>>> scopes;
};

// Helper class to parse boolean expressions from Liberty function attributes
class ExpressionParser {
public:
  ExpressionParser(OpBuilder &builder, StringRef expr,
                   const DenseMap<StringRef, Value> &values)
      : builder(builder), values(values) {
    tokenize(expr);
  }

  Value parse() { return parseOrExpr(); }

private:
  enum class TokenKind { ID, AND, OR, XOR, NOT, LPAREN, RPAREN, END };

  struct Token {
    TokenKind kind;
    StringRef spelling;
  };

  OpBuilder &builder;
  const DenseMap<StringRef, Value> &values;
  SmallVector<Token> tokens;
  size_t pos = 0;

  void tokenize(StringRef expr) {
    const char *ptr = expr.begin();
    const char *end = expr.end();

    while (ptr < end) {
      // Skip whitespace
      if (isspace(*ptr)) {
        ++ptr;
        continue;
      }

      // Identifier
      if (isalnum(*ptr) || *ptr == '_') {
        const char *start = ptr;
        while (ptr < end && (isalnum(*ptr) || *ptr == '_'))
          ++ptr;
        tokens.push_back({TokenKind::ID, StringRef(start, ptr - start)});
        continue;
      }

      // Operators and punctuation
      switch (*ptr) {
      case '*':
      case '&':
        tokens.push_back({TokenKind::AND, StringRef(ptr, 1)});
        break;
      case '+':
      case '|':
        tokens.push_back({TokenKind::OR, StringRef(ptr, 1)});
        break;
      case '^':
        tokens.push_back({TokenKind::XOR, StringRef(ptr, 1)});
        break;
      case '!':
      case '\'':
        tokens.push_back({TokenKind::NOT, StringRef(ptr, 1)});
        break;
      case '(':
        tokens.push_back({TokenKind::LPAREN, StringRef(ptr, 1)});
        break;
      case ')':
        tokens.push_back({TokenKind::RPAREN, StringRef(ptr, 1)});
        break;
      }
      ++ptr;
    }

    tokens.push_back({TokenKind::END, ""});
  }

  Token peek() const { return tokens[pos]; }
  Token consume() { return tokens[pos++]; }

  Value createNot(Value val) {
    Value allOnes = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                               builder.getI1Type(), 1);
    return builder.create<XorOp>(builder.getUnknownLoc(), val, allOnes);
  }

  // Parse: OrExpr -> XorExpr { ('+'|'|') XorExpr }
  Value parseOrExpr() {
    Value lhs = parseXorExpr();
    if (!lhs)
      return nullptr;
    while (peek().kind == TokenKind::OR) {
      consume();
      Value rhs = parseXorExpr();
      if (!rhs)
        return nullptr;
      lhs = builder.create<OrOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  }

  // Parse: XorExpr -> AndExpr { '^' AndExpr }
  Value parseXorExpr() {
    Value lhs = parseAndExpr();
    if (!lhs)
      return nullptr;
    while (peek().kind == TokenKind::XOR) {
      consume();
      Value rhs = parseAndExpr();
      if (!rhs)
        return nullptr;
      lhs = builder.create<XorOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  }

  // Parse: AndExpr -> UnaryExpr { ('*'|'&') UnaryExpr }
  Value parseAndExpr() {
    Value lhs = parseUnaryExpr();
    if (!lhs)
      return nullptr;
    while (peek().kind == TokenKind::AND) {
      consume();
      Value rhs = parseUnaryExpr();
      if (!rhs)
        return nullptr;
      lhs = builder.create<AndOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  }

  // Parse: UnaryExpr -> ('!'|'\'') UnaryExpr | '(' OrExpr ')' ['\''] | ID
  // ['\'']
  Value parseUnaryExpr() {
    // Prefix NOT
    if (peek().kind == TokenKind::NOT) {
      consume();
      Value val = parseUnaryExpr();
      return val ? createNot(val) : nullptr;
    }

    // Parenthesized expression
    if (peek().kind == TokenKind::LPAREN) {
      consume();
      Value val = parseOrExpr();
      if (!val || peek().kind != TokenKind::RPAREN)
        return nullptr;
      consume();
      // Postfix NOT
      if (peek().kind == TokenKind::NOT) {
        consume();
        val = createNot(val);
      }
      return val;
    }

    // Identifier
    if (peek().kind == TokenKind::ID) {
      StringRef name = consume().spelling;
      auto it = values.find(name);
      if (it == values.end())
        return nullptr; // Variable not found
      Value val = it->second;
      // Postfix NOT
      if (peek().kind == TokenKind::NOT) {
        consume();
        val = createNot(val);
      }
      return val;
    }

    return nullptr;
  }
};

class LibertyParser {
public:
  LibertyParser(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                ModuleOp module)
      : lexer(sourceMgr, context), module(module), builder(context) {}

  ParseResult parse();

private:
  LibertyLexer lexer;
  ModuleOp module;
  OpBuilder builder;
  // Map of template name -> vector of variable names (1-based index -> name)
  llvm::StringMap<SmallVector<std::string>> templates;

  ParseResult parseGroup();

  // Specific group parsers
  ParseResult parseLibrary();
  ParseResult parseTemplateDefinition();
  ParseResult parseCell();
  ParseResult parsePin(SmallVectorImpl<PinDef> &pins);
  ParseResult parseScope(SmallVectorImpl<SmallVector<NamedAttribute>> &scopes);
  ParseResult parseGenericGroup(SmallVectorImpl<NamedAttribute> &attrs);

  // Attribute parsing
  ParseResult parseAttribute(Attribute &result);

  // Expression parsing
  Value parseExpression(StringRef expr,
                        const DenseMap<StringRef, Value> &values);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }
  InFlightDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message) {
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
};

} // namespace

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
               !(*curPtr == '*' && *(curPtr + 1) == '/'))
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
         (isalnum(*curPtr) || *curPtr == '_' || *curPtr == '.'))
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
  bool seenDot = false;
  while (curPtr < curBuffer.end()) {
    if (isdigit(*curPtr)) {
      ++curPtr;
    } else if (*curPtr == '.' && !seenDot) {
      seenDot = true;
      ++curPtr;
    } else {
      break;
    }
  }
  return makeToken(LibertyTokenKind::Number, start);
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

  if (isdigit(c) ||
      (c == '.' && curPtr + 1 < curBuffer.end() && isdigit(*(curPtr + 1))))
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
  case '+':
    return makeToken(LibertyTokenKind::Plus, start);
  case '-':
    return makeToken(LibertyTokenKind::Minus, start);
  case '*':
    return makeToken(LibertyTokenKind::Star, start);
  case '/':
    return makeToken(LibertyTokenKind::Slash, start);
  case '!':
    return makeToken(LibertyTokenKind::Exclaim, start);
  case '\'':
    return makeToken(LibertyTokenKind::Apostrophe, start);
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
  while (lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    // Skip any stray tokens that aren't valid group starts
    if (token.kind != LibertyTokenKind::Identifier) {
      lexer.nextToken(); // consume and skip
      continue;
    }
    if (parseGroup())
      return failure();
  }
  return success();
}

ParseResult LibertyParser::parseGroup() {
  auto token = lexer.nextToken();
  if (token.kind != LibertyTokenKind::Identifier)
    return emitError(token.location, "expected group name or attribute");

  // Check if it's a library or cell group
  if (token.spelling == "library")
    return parseLibrary();

  // Consume potential arguments
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    lexer.nextToken(); // (
    while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
      lexer.nextToken(); // consume arg
      if (lexer.peekToken().kind == LibertyTokenKind::Comma)
        lexer.nextToken();
    }
    if (consume(LibertyTokenKind::RParen))
      return failure();
  }

  // Check for block or semicolon
  if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
    // Skip unknown group body
    lexer.nextToken(); // {
    int balance = 1;
    while (balance > 0 &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
      auto t = lexer.nextToken();
      if (t.kind == LibertyTokenKind::LBrace)
        balance++;
      if (t.kind == LibertyTokenKind::RBrace)
        balance--;
    }
    return success();
  }

  if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
    lexer.nextToken(); // :
    // consume value
    while (lexer.peekToken().kind != LibertyTokenKind::Semi &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile)
      lexer.nextToken();
  }

  return consume(LibertyTokenKind::Semi);
}

ParseResult LibertyParser::parseLibrary() {
  if (consume(LibertyTokenKind::LParen))
    return failure();
  auto libName = lexer.nextToken(); // Library name
  if (libName.kind != LibertyTokenKind::Identifier)
    return emitError(libName.location, "expected library name");
  StringRef libNameStr = libName.spelling;
  module->setAttr("liberty.library.name", builder.getStringAttr(libNameStr));
  if (consume(LibertyTokenKind::RParen) || consume(LibertyTokenKind::LBrace))
    return failure();

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::Identifier &&
        token.spelling == "cell") {
      if (parseCell())
        return failure();
    } else if (token.kind == LibertyTokenKind::Identifier &&
               (token.spelling == "lu_table_template" ||
                token.spelling == "power_lut_template" ||
                token.spelling == "lu_table_template")) {
      if (parseTemplateDefinition())
        return failure();
    } else {
      // Skip other library attributes/groups
      auto groupName = lexer.nextToken(); // identifier
      if (groupName.kind != LibertyTokenKind::Identifier)
        return emitError(groupName.location, "expected attribute name");

      // Handle arguments like group(arg)
      if (lexer.peekToken().is(LibertyTokenKind::LParen)) {
        lexer.nextToken(); // (
        while (!lexer.peekToken().is(LibertyTokenKind::RParen) &&
               !lexer.peekToken().is(LibertyTokenKind::EndOfFile)) {
          lexer.nextToken();
          if (lexer.peekToken().is(LibertyTokenKind::Comma))
            lexer.nextToken();
        }
        if (expect(LibertyTokenKind::RParen))
          return failure();
        lexer.nextToken(); // )
      }

      if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
        lexer.nextToken(); // {
        int balance = 1;
        while (balance > 0 &&
               lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
          auto t = lexer.nextToken();
          if (t.kind == LibertyTokenKind::LBrace)
            balance++;
          if (t.kind == LibertyTokenKind::RBrace)
            balance--;
        }
      } else if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
        lexer.nextToken(); // :
        while (lexer.peekToken().kind != LibertyTokenKind::Semi)
          lexer.nextToken();
        if (expect(LibertyTokenKind::Semi))
          return failure();
        lexer.nextToken(); // ;
      } else if (lexer.peekToken().kind == LibertyTokenKind::Semi) {
        lexer.nextToken(); // ;
      }
    }
  }

  return consume(LibertyTokenKind::RBrace, "expected '}'");
}

// Parse a template like: lu_table_template (delay_template_6x6) { variable_1:
// total_output_net_capacitance; variable_2: input_net_transition; }
ParseResult LibertyParser::parseTemplateDefinition() {
  // We are at the identifier token for the template name (not consumed yet).
  lexer.nextToken(); // consume template-kind identifier (e.g.,
                     // 'lu_table_template')
  // Expect (template_name)
  if (consume(LibertyTokenKind::LParen))
    return failure();
  auto templTok = lexer.nextToken();
  StringRef templName = templTok.spelling;
  if (templTok.kind == LibertyTokenKind::String)
    templName = templName.drop_front().drop_back();
  if (consume(LibertyTokenKind::RParen) || consume(LibertyTokenKind::LBrace))
    return failure();

  // We'll collect variable_N names. Use a sparse vector sized by index.
  SmallVector<std::string> varNames;

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto tok = lexer.nextToken();
    if (tok.kind != LibertyTokenKind::Identifier)
      continue;
    StringRef id = tok.spelling; // e.g., variable_1
    if (id.starts_with("variable_")) {
      if (consume(LibertyTokenKind::Colon))
        return failure();
      auto valTok = lexer.nextToken();
      if (valTok.kind == LibertyTokenKind::Identifier ||
          valTok.kind == LibertyTokenKind::String) {
        StringRef varName = valTok.spelling;
        if (valTok.kind == LibertyTokenKind::String)
          varName = varName.drop_front().drop_back();
        StringRef idxStr = id.substr(strlen("variable_"));
        unsigned idx = 0;
        if (!idxStr.getAsInteger(10, idx)) {
          if (idx >= varNames.size())
            varNames.resize(idx + 1);
          varNames[idx] = varName.str();
        }
      }
      // skip to semicolon
      while (lexer.peekToken().kind != LibertyTokenKind::Semi &&
             lexer.peekToken().kind != LibertyTokenKind::EndOfFile)
        lexer.nextToken();
      if (lexer.peekToken().kind == LibertyTokenKind::Semi)
        lexer.nextToken();
      continue;
    }

    // Skip other items inside template body (index_*, etc.)
    if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
      lexer.nextToken();
      while (lexer.peekToken().kind != LibertyTokenKind::RParen &&
             lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
        lexer.nextToken();
        if (lexer.peekToken().kind == LibertyTokenKind::Comma)
          lexer.nextToken();
      }
      if (lexer.peekToken().kind == LibertyTokenKind::RParen)
        lexer.nextToken();
    }
    if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
      lexer.nextToken();
      int bal = 1;
      while (bal > 0 && lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
        auto t = lexer.nextToken();
        if (t.kind == LibertyTokenKind::LBrace)
          bal++;
        if (t.kind == LibertyTokenKind::RBrace)
          bal--;
      }
    }
  }

  if (!templName.empty())
    templates[templName] = std::move(varNames);
  // Also attach a module attribute for debugging/visibility so downstream
  // inspection can see what templates were recorded.
  SmallVector<Attribute> varAttrs;
  for (auto &s : templates[templName])
    varAttrs.push_back(builder.getStringAttr(s));
  module->setAttr(("liberty.template." + templName).str(),
                  builder.getArrayAttr(varAttrs));

  if (consume(LibertyTokenKind::RBrace))
    return failure();
  return success();
}

ParseResult LibertyParser::parseCell() {
  lexer.nextToken(); // consume 'cell'
  if (consume(LibertyTokenKind::LParen))
    return failure();
  auto nameToken = lexer.nextToken();
  StringRef cellName = nameToken.spelling;
  if (nameToken.kind == LibertyTokenKind::String)
    cellName = cellName.drop_front().drop_back();

  if (consume(LibertyTokenKind::RParen) || consume(LibertyTokenKind::LBrace))
    return failure();

  SmallVector<PinDef> pins;
  SmallVector<NamedAttribute> cellAttrs;

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::Identifier && token.spelling == "pin") {
      if (parsePin(pins))
        return failure();
    } else {
      // Capture cell attributes like area
      auto attrToken = lexer.nextToken(); // identifier
      StringRef attrName = attrToken.spelling;

      // Check if it's a simple attribute (name : value ;)
      if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
        lexer.nextToken(); // :
        auto valueToken = lexer.peekToken();
        if (valueToken.kind == LibertyTokenKind::Number ||
            valueToken.kind == LibertyTokenKind::String ||
            valueToken.kind == LibertyTokenKind::Identifier) {
          lexer.nextToken();
          StringRef strValue = valueToken.spelling;
          if (valueToken.kind == LibertyTokenKind::String)
            strValue = strValue.drop_front().drop_back();
          cellAttrs.push_back(
              builder.getNamedAttr(attrName, builder.getStringAttr(strValue)));
          while (lexer.peekToken().kind != LibertyTokenKind::Semi)
            lexer.nextToken();
          if (consume(LibertyTokenKind::Semi))
            return failure();
          continue;
        }
      }

      // Skip complex attributes/groups

      if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
        lexer.nextToken(); // (
        while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
          lexer.nextToken();
          if (lexer.peekToken().kind == LibertyTokenKind::Comma)
            lexer.nextToken();
        }
        if (consume(LibertyTokenKind::RParen, "expected ')'"))
          return failure();
      }

      if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
        lexer.nextToken(); // {
        int balance = 1;
        while (balance > 0 &&
               lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
          auto t = lexer.nextToken();
          if (t.kind == LibertyTokenKind::LBrace)
            balance++;
          if (t.kind == LibertyTokenKind::RBrace)
            balance--;
        }
      } else {
        if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
          lexer.nextToken(); // :
          while (lexer.peekToken().kind != LibertyTokenKind::Semi)
            lexer.nextToken();
        }
        if (consume(LibertyTokenKind::Semi, "expected ';'"))
          return failure();
      }
    }
  }

  if (consume(LibertyTokenKind::RBrace, "expected '}'"))
    return failure();

  // Convert PinDefs to PortInfos
  SmallVector<PortInfo> ports;
  for (const auto &pin : pins) {
    if (pin.isInput || pin.isOutput) {
      PortInfo port;
      port.name = builder.getStringAttr(pin.name);
      port.type = builder.getI1Type(); // Assume 1-bit
      port.dir = pin.isInput ? ModulePort::Direction::Input
                             : ModulePort::Direction::Output;
      // Set pin attributes
      SmallVector<NamedAttribute> pinAttrs = pin.attrs;
      for (const auto &scope : pin.scopes) {
        SmallVector<Attribute> scopeAttrs;
        for (const auto &s : scope.second) {
          scopeAttrs.push_back(builder.getDictionaryAttr(s));
        }
        pinAttrs.push_back(
            builder.getNamedAttr(("liberty." + scope.first()).str(),
                                 builder.getArrayAttr(scopeAttrs)));
      }

      if (!pinAttrs.empty()) {
        port.attrs = builder.getDictionaryAttr(pinAttrs);
      }
      ports.push_back(port);
    }
  }

  // Create HWModule with cell attributes
  builder.setInsertionPointToStart(module.getBody());
  auto hwMod = HWModuleOp::create(
      builder, builder.getUnknownLoc(), builder.getStringAttr(cellName), ports,
      ArrayAttr{},            // parameters
      cellAttrs, StringAttr{} // cell attributes as module attributes
  );

  // HWModuleOp creates a block with ports.
  if (hwMod.getBody().empty()) {
    auto *block = builder.createBlock(&hwMod.getBody());
    builder.setInsertionPointToStart(block);
  } else {
    builder.setInsertionPointToStart(&hwMod.getBody().front());
  }

  // Map inputs to values
  DenseMap<StringRef, Value> values;
  int argIndex = 0;
  for (const auto &pin : pins) {
    if (pin.isInput) {
      values[pin.name] = hwMod.getBody().front().getArgument(argIndex++);
    }
  }

  // Generate outputs
  SmallVector<Value> outputValues;
  for (const auto &pin : pins) {
    if (pin.isOutput) {
      if (!pin.function.empty()) {
        Value val = parseExpression(pin.function, values);
        if (val) {
          outputValues.push_back(val);
        } else {
          // Fallback to unknown/constant 0 if parsing fails
          outputValues.push_back(builder.create<ConstantOp>(
              builder.getUnknownLoc(), builder.getI1Type(), 0));
        }
      } else {
        // No function, just output 0
        outputValues.push_back(builder.create<ConstantOp>(
            builder.getUnknownLoc(), builder.getI1Type(), 0));
      }
    }
  }
  auto *outputOp = hwMod.getBodyBlock()->getTerminator();
  outputOp->setOperands(outputValues);

  return success();
}

ParseResult LibertyParser::parsePin(SmallVectorImpl<PinDef> &pins) {
  lexer.nextToken(); // consume 'pin'
  if (consume(LibertyTokenKind::LParen, "expected '('"))
    return failure();
  auto nameToken = lexer.nextToken();
  StringRef pinName = nameToken.spelling;
  if (nameToken.kind == LibertyTokenKind::String)
    pinName = pinName.drop_front().drop_back();
  if (consume(LibertyTokenKind::RParen, "expected ')'"))
    return failure();
  if (consume(LibertyTokenKind::LBrace, "expected '{'"))
    return failure();

  PinDef pin;
  pin.name = pinName;

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.nextToken();
    if (token.kind == LibertyTokenKind::Identifier &&
        token.spelling == "direction") {
      if (consume(LibertyTokenKind::Colon, "expected ':'"))
        return failure();
      auto dirToken = lexer.nextToken();
      if (dirToken.spelling == "input")
        pin.isInput = true;
      else if (dirToken.spelling == "output")
        pin.isOutput = true;
      if (consume(LibertyTokenKind::Semi, "expected ';'"))
        return failure();
    } else if (token.kind == LibertyTokenKind::Identifier &&
               token.spelling == "function") {
      if (consume(LibertyTokenKind::Colon, "expected ':'"))
        return failure();
      auto funcToken = lexer.nextToken();
      if (funcToken.kind == LibertyTokenKind::String) {
        pin.function = funcToken.spelling.drop_front().drop_back().str();
      } else {
        // Handle unquoted function?
        pin.function = funcToken.spelling.str();
      }
      if (consume(LibertyTokenKind::Semi, "expected ';'"))
        return failure();
    } else if (token.kind == LibertyTokenKind::Identifier &&
               token.spelling == "timing") {
      if (parseScope(pin.scopes["timing"]))
        return failure();
    } else {
      // Capture other pin attributes
      StringRef attrName = token.spelling;

      // Check for simple attributes (name : value ;)
      if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
        lexer.nextToken(); // :
        auto valueToken = lexer.peekToken();
        if (valueToken.kind == LibertyTokenKind::Number ||
            valueToken.kind == LibertyTokenKind::String ||
            valueToken.kind == LibertyTokenKind::Identifier) {
          lexer.nextToken();
          StringRef strValue = valueToken.spelling;
          if (valueToken.kind == LibertyTokenKind::String)
            strValue = strValue.drop_front().drop_back();
          pin.attrs.push_back(
              builder.getNamedAttr(attrName, builder.getStringAttr(strValue)));
          while (lexer.peekToken().kind != LibertyTokenKind::Semi)
            lexer.nextToken();
          if (consume(LibertyTokenKind::Semi, "expected ';'"))
            return failure();
          continue;
        }
      }

      // Skip complex attributes/groups
      if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
        lexer.nextToken(); // (
        while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
          lexer.nextToken();
          if (lexer.peekToken().kind == LibertyTokenKind::Comma)
            lexer.nextToken();
        }
        if (consume(LibertyTokenKind::RParen, "expected ')'"))
          return failure();
        if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
          lexer.nextToken(); // {
          int balance = 1;
          while (balance > 0) {
            auto t = lexer.nextToken();
            if (t.kind == LibertyTokenKind::LBrace)
              balance++;
            if (t.kind == LibertyTokenKind::RBrace)
              balance--;
          }
        }
      }

      // Skip remaining tokens until semicolon
      while (lexer.peekToken().kind != LibertyTokenKind::Semi &&
             lexer.peekToken().kind != LibertyTokenKind::RBrace &&
             lexer.peekToken().kind != LibertyTokenKind::EndOfFile)
        lexer.nextToken();
      if (lexer.peekToken().kind == LibertyTokenKind::Semi)
        lexer.nextToken();
    }
  }

  if (consume(LibertyTokenKind::RBrace, "expected '}'"))
    return failure();

  if (pin.isInput || pin.isOutput) {
    pins.push_back(pin);
  }

  return success();
}

ParseResult LibertyParser::parseScope(
    SmallVectorImpl<SmallVector<NamedAttribute>> &scopes) {
  if (consume(LibertyTokenKind::LParen, "expected '('"))
    return failure();
  // timing() usually has no args, but let's handle potential args or empty
  while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
    lexer.nextToken(); // consume arg
    if (lexer.peekToken().kind == LibertyTokenKind::Comma)
      lexer.nextToken();
  }
  if (consume(LibertyTokenKind::RParen) || consume(LibertyTokenKind::LBrace))
    return failure();

  SmallVector<NamedAttribute> scope;
  if (parseGenericGroup(scope))
    return failure();

  if (consume(LibertyTokenKind::RBrace))
    return failure();

  scopes.push_back(std::move(scope));
  return success();
}

ParseResult
LibertyParser::parseGenericGroup(SmallVectorImpl<NamedAttribute> &attrs) {
  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.nextToken();
    if (token.kind == LibertyTokenKind::Identifier) {
      StringRef attrName = token.spelling;
      if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
        lexer.nextToken(); // :
        Attribute attrValue;
        if (parseAttribute(attrValue))
          return failure();
        attrs.push_back(builder.getNamedAttr(attrName, attrValue));
        if (consume(LibertyTokenKind::Semi, "expected ';'"))
          return failure();
      } else if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
        // group
        lexer.nextToken(); // (
        SmallVector<Attribute> args;
        while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
          Attribute arg;
          if (parseAttribute(arg))
            return failure();
          args.push_back(arg);
          if (lexer.peekToken().kind == LibertyTokenKind::Comma)
            lexer.nextToken();
        }
        if (consume(LibertyTokenKind::RParen, "expected ')'"))
          return failure();

        // Parse body if present
        if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
          lexer.nextToken(); // {
          SmallVector<NamedAttribute> nestedAttrs;
          if (!args.empty())
            nestedAttrs.push_back(
                builder.getNamedAttr("args", builder.getArrayAttr(args)));

          if (parseGenericGroup(nestedAttrs))
            return failure();

          if (consume(LibertyTokenKind::RBrace, "expected '}'"))
            return failure();

          // If args referenced a template we recorded earlier, inline it by
          // remapping index_N -> variable_N names. Prefer extracting the
          // template name from the nestedAttrs 'args' entry to avoid
          // dependence on the outer 'args' vector.
          StringRef templName;
          bool haveTemplate = false;
          for (auto &na : nestedAttrs) {
            if (na.getName().getValue() == "args") {
              if (auto arr = dyn_cast<ArrayAttr>(na.getValue())) {
                if (!arr.getValue().empty()) {
                  if (auto s = dyn_cast<StringAttr>(arr.getValue()[0])) {
                    templName = s.getValue();
                    haveTemplate = true;
                  }
                }
              }
              break;
            }
          }
          if (haveTemplate) {
            auto it = templates.find(templName);
            if (it != templates.end()) {
              SmallVector<NamedAttribute> remapped;
              for (auto &na : nestedAttrs) {
                StringRef n = na.getName().getValue();
                if (n == "args")
                  continue; // drop the args member
                // remap index_N
                if (n.starts_with("index_")) {
                  StringRef idxStr = n.substr(strlen("index_"));
                  unsigned idx = 0;
                  if (!idxStr.getAsInteger(10, idx)) {
                    // templates vector uses 0-based with idx == N
                    if (idx < it->second.size() && !it->second[idx].empty()) {
                      remapped.push_back(
                          builder.getNamedAttr(it->second[idx], na.getValue()));
                      continue;
                    }
                  }
                }
                // default: keep original
                remapped.push_back(na);
              }
              attrs.push_back(builder.getNamedAttr(
                  attrName, builder.getDictionaryAttr(remapped)));
              continue;
            }
          }

          attrs.push_back(builder.getNamedAttr(
              attrName, builder.getDictionaryAttr(nestedAttrs)));
        } else {
          // No body?
          // If args is empty, it's just name(); which is weird but possible?
          // If args is not empty, it's name(args);
          // We can store it as name = [args]
          attrs.push_back(
              builder.getNamedAttr(attrName, builder.getArrayAttr(args)));
          // Consume optional semicolon for attributes like index_1(...);
          if (lexer.peekToken().kind == LibertyTokenKind::Semi)
            lexer.nextToken();
        }
      }
    }
  }
  return success();
}

// Parse an attribute value, which can be:
// - A string: "value"
// - A number: 1.23
// - An identifier: value
// - A list: "val1, val2, val3" (comma-separated values)
ParseResult LibertyParser::parseAttribute(Attribute &result) {
  auto token = lexer.peekToken();

  // Check for quoted string with comma-separated values (array)
  if (token.is(LibertyTokenKind::String)) {
    lexer.nextToken();
    StringRef str = token.spelling.drop_front().drop_back();

    // Check if it contains commas (array of values)
    if (str.contains(',')) {
      SmallVector<Attribute> elements;
      SmallVector<StringRef> parts;
      str.split(parts, ',');

      for (StringRef part : parts) {
        part = part.trim();
        // Try to parse as float
        double val;
        if (!part.getAsDouble(val)) {
          elements.push_back(builder.getF64FloatAttr(val));
        } else {
          // Keep as string if not a valid number
          elements.push_back(builder.getStringAttr(part));
        }
      }
      result = builder.getArrayAttr(elements);
      return success();
    }

    // Single string value - try to parse as number first
    double val;
    if (!str.getAsDouble(val)) {
      result = builder.getF64FloatAttr(val);
      return success();
    }
    result = builder.getStringAttr(str);
    return success();
  }

  // Number token
  if (token.is(LibertyTokenKind::Number)) {
    lexer.nextToken();
    StringRef numStr = token.spelling;
    double val;
    if (!numStr.getAsDouble(val)) {
      result = builder.getF64FloatAttr(val);
      return success();
    }
    // Fallback to string if parsing fails
    result = builder.getStringAttr(numStr);
    return success();
  }

  // Identifier token
  if (token.is(LibertyTokenKind::Identifier)) {
    lexer.nextToken();
    result = builder.getStringAttr(token.spelling);
    return success();
  }

  return emitError(token.location, "expected attribute value");
}

// Parse boolean expressions from Liberty function attributes
// Supports: *, +, ^, !, (), and identifiers
Value LibertyParser::parseExpression(StringRef expr,
                                     const DenseMap<StringRef, Value> &values) {
  ExpressionParser parser(builder, expr, values);
  return parser.parse();
}

namespace circt {
void registerImportLibertyTranslation() {
  TranslateToMLIRRegistration reg(
      "import-liberty", "Import Liberty file",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        ModuleOp module = ModuleOp::create(UnknownLoc::get(context));
        LibertyParser parser(sourceMgr, context, module);
        // Load required dialects
        context->loadDialect<hw::HWDialect>();
        context->loadDialect<comb::CombDialect>();
        if (failed(parser.parse()))
          return OwningOpRef<ModuleOp>();
        return OwningOpRef<ModuleOp>(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<HWDialect>();
        registry.insert<comb::CombDialect>();
      });
}
} // namespace circt
