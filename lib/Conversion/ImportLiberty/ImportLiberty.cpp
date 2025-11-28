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

struct LibertyToken {
  LibertyTokenKind kind;
  StringRef spelling;
  SMLoc location;

  LibertyToken(LibertyTokenKind kind, StringRef spelling, SMLoc location)
      : kind(kind), spelling(spelling), location(location) {}
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
};

class LibertyParser {
public:
  LibertyParser(const llvm::SourceMgr &sourceMgr, MLIRContext *context,
                ModuleOp module)
      : lexer(sourceMgr, context), context(context), module(module),
        builder(context) {}

  ParseResult parse();

private:
  LibertyLexer lexer;
  MLIRContext *context;
  ModuleOp module;
  OpBuilder builder;

  ParseResult parseGroup();

  // Specific group parsers
  ParseResult parseLibrary();
  ParseResult parseCell();
  ParseResult parsePin(SmallVectorImpl<PinDef> &pins);

  // Expression parsing
  Value parseExpression(StringRef expr,
                        const DenseMap<StringRef, Value> &values);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) {
    return mlir::emitError(lexer.translateLocation(loc), message);
  }

  ParseResult consume(LibertyTokenKind kind, const Twine &msg) {
    if (lexer.nextToken().kind != kind)
      return emitError(lexer.getCurrentLoc(), msg);
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
    if (consume(LibertyTokenKind::RParen, "expected ')'"))
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

  return consume(LibertyTokenKind::Semi, "expected ';'");
}

ParseResult LibertyParser::parseLibrary() {
  if (consume(LibertyTokenKind::LParen, "expected '(' after library"))
    return failure();
  (void)lexer.nextToken(); // Library name
  if (consume(LibertyTokenKind::RParen, "expected ')'"))
    return failure();
  if (consume(LibertyTokenKind::LBrace, "expected '{'"))
    return failure();

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::Identifier &&
        token.spelling == "cell") {
      if (parseCell())
        return failure();
    } else {
      // Skip other library attributes/groups
      (void)lexer.nextToken(); // identifier

      // Handle arguments like group(arg)
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

  return consume(LibertyTokenKind::RBrace, "expected '}'");
}

ParseResult LibertyParser::parseCell() {
  lexer.nextToken(); // consume 'cell'
  if (consume(LibertyTokenKind::LParen, "expected '('"))
    return failure();
  auto nameToken = lexer.nextToken();
  StringRef cellName = nameToken.spelling;
  if (nameToken.kind == LibertyTokenKind::String)
    cellName = cellName.drop_front().drop_back();

  if (consume(LibertyTokenKind::RParen, "expected ')'"))
    return failure();
  if (consume(LibertyTokenKind::LBrace, "expected '{'"))
    return failure();

  SmallVector<PinDef> pins;

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::Identifier && token.spelling == "pin") {
      if (parsePin(pins))
        return failure();
    } else {
      // Skip other cell attributes
      (void)lexer.nextToken(); // identifier

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
      ports.push_back(port);
    }
  }

  // Create HWModule
  builder.setInsertionPointToStart(module.getBody());
  auto hwMod = HWModuleOp::create(
      builder, builder.getUnknownLoc(), builder.getStringAttr(cellName), ports,
      ArrayAttr{},                             // parameters
      ArrayRef<NamedAttribute>{}, StringAttr{} // comment
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
    } else {
      // Skip attribute value
      if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
        lexer.nextToken(); // :
        while (lexer.peekToken().kind != LibertyTokenKind::Semi)
          lexer.nextToken();
      } else if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
        // Skip complex attribute/group
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

// Simple expression parser
// Supports: *, +, ^, !, (), and identifiers
Value LibertyParser::parseExpression(StringRef expr,
                                     const DenseMap<StringRef, Value> &values) {
  // This is a very simple recursive descent parser for boolean expressions
  // Grammar:
  // Expr -> Term { '+' Term }   (OR)
  // Term -> Factor { '*' Factor } (AND)
  // Factor -> Atom { '^' Atom } (XOR)
  // Atom -> '!' Atom | '(' Expr ')' | Identifier

  // Tokenizer for expression
  struct ExprToken {
    enum Kind { ID, AND, OR, XOR, NOT, LPAREN, RPAREN, END, ERR };
    Kind kind;
    StringRef spelling;
  };

  SmallVector<ExprToken> tokens;
  const char *ptr = expr.begin();
  const char *end = expr.end();

  while (ptr < end) {
    if (isspace(*ptr)) {
      ++ptr;
      continue;
    }
    if (isalnum(*ptr) || *ptr == '_') {
      const char *start = ptr;
      while (ptr < end && (isalnum(*ptr) || *ptr == '_'))
        ++ptr;
      tokens.push_back({ExprToken::ID, StringRef(start, ptr - start)});
    } else {
      switch (*ptr) {
      case '*':
        tokens.push_back({ExprToken::AND, "*"});
        break;
      case '&':
        tokens.push_back({ExprToken::AND, "&"});
        break; // Support & as AND too
      case '+':
        tokens.push_back({ExprToken::OR, "+"});
        break;
      case '|':
        tokens.push_back({ExprToken::OR, "|"});
        break; // Support | as OR too
      case '^':
        tokens.push_back({ExprToken::XOR, "^"});
        break;
      case '!':
        tokens.push_back({ExprToken::NOT, "!"});
        break;
      case '\'':
        tokens.push_back({ExprToken::NOT, "'"});
        break; // Postfix NOT not fully supported in this simple parser,
               // treating as prefix error or ignore?
      case '(':
        tokens.push_back({ExprToken::LPAREN, "("});
        break;
      case ')':
        tokens.push_back({ExprToken::RPAREN, ")"});
        break;
      default:
        break; // Ignore unknown chars
      }
      ++ptr;
    }
  }
  tokens.push_back({ExprToken::END, ""});

  int pos = 0;
  auto peek = [&]() { return tokens[pos]; };
  auto next = [&]() { return tokens[pos++]; };

  std::function<Value()> parseExpr;
  std::function<Value()> parseTerm;
  std::function<Value()> parseFactor;
  std::function<Value()> parseAtom;

  parseExpr = [&]() -> Value {
    Value lhs = parseTerm();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::OR) {
      next();
      Value rhs = parseTerm();
      if (!rhs)
        return nullptr;
      lhs = builder.create<OrOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseTerm = [&]() -> Value {
    Value lhs = parseFactor();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::AND) {
      next();
      Value rhs = parseFactor();
      if (!rhs)
        return nullptr;
      lhs = builder.create<AndOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseFactor = [&]() -> Value {
    Value lhs = parseAtom();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::XOR) {
      next();
      Value rhs = parseAtom();
      if (!rhs)
        return nullptr;
      lhs = builder.create<XorOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseAtom = [&]() -> Value {
    if (peek().kind == ExprToken::NOT) {
      next();
      Value val = parseAtom();
      if (!val)
        return nullptr;
      // create constant -1 (all ones) for XOR
      Value allOnes = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                                 builder.getI1Type(), 1);
      return builder.create<XorOp>(builder.getUnknownLoc(), val, allOnes);
    }
    if (peek().kind == ExprToken::LPAREN) {
      next();
      Value val = parseExpr();
      if (peek().kind != ExprToken::RPAREN)
        return nullptr;
      next();
      return val;
    }
    if (peek().kind == ExprToken::ID) {
      StringRef name = next().spelling;
      auto it = values.find(name);
      if (it != values.end())
        return it->second;
      return nullptr; // Variable not found
    }
    return nullptr;
  };

  return parseExpr();
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
