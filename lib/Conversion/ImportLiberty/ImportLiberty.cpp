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
  if (consume(LibertyTokenKind::RParen, "expected ')'"))
    return failure();
  if (consume(LibertyTokenKind::LBrace, "expected '{'"))
    return failure();

  SmallVector<NamedAttribute> scope;
  if (parseGenericGroup(scope))
    return failure();

  if (consume(LibertyTokenKind::RBrace, "expected '}'"))
    return failure();

  scopes.push_back(scope);
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
        auto val = lexer.nextToken();
        StringRef valStr = val.spelling;
        if (val.kind == LibertyTokenKind::String)
          valStr = valStr.drop_front().drop_back();
        attrs.push_back(
            builder.getNamedAttr(attrName, builder.getStringAttr(valStr)));
        if (consume(LibertyTokenKind::Semi, "expected ';'"))
          return failure();
      } else if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
        // group
        lexer.nextToken(); // (
        SmallVector<Attribute> args;
        while (lexer.peekToken().kind != LibertyTokenKind::RParen) {
          auto arg = lexer.nextToken();
          StringRef argStr = arg.spelling;
          if (arg.kind == LibertyTokenKind::String)
            argStr = argStr.drop_front().drop_back();
          args.push_back(builder.getStringAttr(argStr));
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
  std::function<Value()> parseXorExpr;
  std::function<Value()> parseTerm;
  std::function<Value()> parseAtom;

  parseExpr = [&]() -> Value {
    Value lhs = parseXorExpr();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::OR) {
      next();
      Value rhs = parseXorExpr();
      if (!rhs)
        return nullptr;
      lhs = builder.create<OrOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseXorExpr = [&]() -> Value {
    Value lhs = parseTerm();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::XOR) {
      next();
      Value rhs = parseTerm();
      if (!rhs)
        return nullptr;
      lhs = builder.create<XorOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseTerm = [&]() -> Value {
    Value lhs = parseAtom();
    if (!lhs)
      return nullptr;
    while (peek().kind == ExprToken::AND) {
      next();
      Value rhs = parseAtom();
      if (!rhs)
        return nullptr;
      lhs = builder.create<AndOp>(builder.getUnknownLoc(), lhs, rhs);
    }
    return lhs;
  };

  parseAtom = [&]() -> Value {
    // Prefix NOT
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

    // Parenthesized expression
    if (peek().kind == ExprToken::LPAREN) {
      next();
      Value val = parseExpr();
      if (peek().kind != ExprToken::RPAREN)
        return nullptr;
      next();
      // Check for postfix NOT
      if (peek().kind == ExprToken::NOT) {
        next();
        Value allOnes = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                                   builder.getI1Type(), 1);
        val = builder.create<XorOp>(builder.getUnknownLoc(), val, allOnes);
      }
      return val;
    }

    // Identifier
    if (peek().kind == ExprToken::ID) {
      StringRef name = next().spelling;
      auto it = values.find(name);
      if (it == values.end())
        return nullptr; // Variable not found
      Value val = it->second;
      // Check for postfix NOT
      if (peek().kind == ExprToken::NOT) {
        next();
        Value allOnes = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                                   builder.getI1Type(), 1);
        val = builder.create<XorOp>(builder.getUnknownLoc(), val, allOnes);
      }
      return val;
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
