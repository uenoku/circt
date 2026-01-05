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
#include "circt/Dialect/Synth/SynthDialect.h"
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
#include "llvm/ADT/StringSwitch.h"
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

// Parsed result of a Liberty group
struct LibertyGroup {
  StringRef name;
  SMLoc loc;
  SmallVector<Attribute> args;
  SmallVector<std::pair<StringRef, Attribute>> attrs;
  SmallVector<std::unique_ptr<LibertyGroup>> subGroups;

  // Helper to find a subgroup by name
  const LibertyGroup *findGroup(StringRef name) const {
    for (const auto &g : subGroups)
      if (g->name == name)
        return g.get();
    return nullptr;
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
  // Map of template name -> vector of variable names (1-based index -> name)
  llvm::StringMap<SmallVector<std::string>> templates;

  ParseResult parseGroup();

  // Specific group parsers
  ParseResult parseLibrary();
  ParseResult parseGroup(StringRef name, Location loc, LibertyGroup &group);
  ParseResult parseGroupBody(LibertyGroup &group);
  ParseResult parseComplexAttribute(LibertyGroup &parent);
  ParseResult parseGroup(LibertyGroup &parent);
  ParseResult parseStatement(LibertyGroup &parent);

  // Lowering methods
  ParseResult lowerCell(const LibertyGroup &group);
  ParseResult lowerTemplate(const LibertyGroup &group);

  //===--------------------------------------------------------------------===//
  // Parser for Subgroup of "cell" group.
  //===--------------------------------------------------------------------===//

  // Parse "timing".
  Attribute lowerTimingGroup(const LibertyGroup &group);
  // Parse "internal_power"
  Attribute lowerInternalPowerGroup(const LibertyGroup &group);
  // Parse "output_current_rise"
  Attribute lowerOutputCurrentRiseGroup(const LibertyGroup &group);
  // Parse "output_current_fall"
  Attribute lowerOutputCurrentFallGroup(const LibertyGroup &group);
  // Parse "receiver_capacitance"
  Attribute lowerReceiverCapacitanceGroup(const LibertyGroup &group);

  Attribute convertGroupToAttr(const LibertyGroup &group);

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

  // Helper to skip balanced { ... }
  void skipBlock() {
    lexer.nextToken(); // consume '{'
    int balance = 1;
    while (balance > 0 &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
      auto t = lexer.nextToken();
      if (t.kind == LibertyTokenKind::LBrace)
        balance++;
      if (t.kind == LibertyTokenKind::RBrace)
        balance--;
    }
  }

  // Helper to skip ( ... )
  ParseResult skipArguments() {
    lexer.nextToken(); // consume '('
    while (lexer.peekToken().kind != LibertyTokenKind::RParen &&
           lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
      lexer.nextToken();
      if (lexer.peekToken().kind == LibertyTokenKind::Comma)
        lexer.nextToken();
    }
    return consume(LibertyTokenKind::RParen, "expected ')'");
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
    if (skipArguments())
      return failure();
  }

  // Check for block or semicolon
  if (lexer.peekToken().kind == LibertyTokenKind::LBrace) {
    skipBlock();
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
  if (consume(LibertyTokenKind::RParen) || consume(LibertyTokenKind::LBrace))
    return failure();

  SmallVector<NamedAttribute> libraryAttrs;
  libraryAttrs.push_back(
      builder.getNamedAttr("name", builder.getStringAttr(libNameStr)));

  while (lexer.peekToken().kind != LibertyTokenKind::RBrace &&
         lexer.peekToken().kind != LibertyTokenKind::EndOfFile) {
    auto token = lexer.peekToken();
    if (token.kind == LibertyTokenKind::Identifier) {
      if (token.spelling == "cell") {
        lexer.nextToken(); // consume 'cell'
        LibertyGroup cellGroup;
        cellGroup.name = "cell";
        cellGroup.loc = token.location;
        if (parseGroupBody(cellGroup))
          return failure();
        if (lowerCell(cellGroup))
          return failure();
        continue;
      }
      if (token.spelling.ends_with("_template")) {
        StringRef name = token.spelling;
        lexer.nextToken(); // consume template kind
        LibertyGroup templGroup;
        templGroup.name = name;
        templGroup.loc = token.location;
        if (parseGroupBody(templGroup))
          return failure();
        if (lowerTemplate(templGroup))
          return failure();
        continue;
      }
    }

    // Generic statement (attribute or other group)
    LibertyGroup dummyParent;
    if (parseStatement(dummyParent))
      return failure();

    // Add attributes to module
    for (auto &attr : dummyParent.attrs) {
      libraryAttrs.push_back(builder.getNamedAttr(attr.first, attr.second));
    }
    // Add subgroups to module? Usually library level groups are ignored or
    // handled specifically.
    // For now, we can ignore other groups or add them as attributes if needed.
  }

  if (!libraryAttrs.empty()) {
    // Check duplicate.
    llvm::DenseSet<StringAttr> seenNames;
    for (auto &attr : libraryAttrs) {
      if (!seenNames.insert(attr.getName()).second) {
        return emitError(libName.location, "duplicate library attribute: ")
               << attr.getName();
      }
    }
    module->setAttr("synth.liberty.library",
                    builder.getDictionaryAttr(libraryAttrs));
  }

  return consume(LibertyTokenKind::RBrace, "expected '}'");
}

// Parse a template like: lu_table_template (delay_template_6x6) { variable_1:
// total_output_net_capacitance; variable_2: input_net_transition; }
Attribute LibertyParser::convertGroupToAttr(const LibertyGroup &group) {
  SmallVector<NamedAttribute> attrs;
  if (!group.args.empty())
    attrs.push_back(
        builder.getNamedAttr("args", builder.getArrayAttr(group.args)));

  for (const auto &attr : group.attrs)
    attrs.push_back(builder.getNamedAttr(attr.first, attr.second));

  llvm::StringMap<SmallVector<Attribute>> subGroups;
  for (const auto &sub : group.subGroups)
    subGroups[sub->name].push_back(convertGroupToAttr(*sub));

  for (auto &it : subGroups)
    attrs.push_back(
        builder.getNamedAttr(it.getKey(), builder.getArrayAttr(it.getValue())));

  return builder.getDictionaryAttr(attrs);
}

ParseResult LibertyParser::lowerTemplate(const LibertyGroup &group) {
  if (group.args.empty())
    return success();

  StringRef templateName;
  if (auto strAttr = dyn_cast<StringAttr>(group.args[0]))
    templateName = strAttr.getValue();
  else
    return success();

  // Simple implementation: extract variable names
  SmallVector<std::string> vars;
  for (const auto &attr : group.attrs) {
    // Prefix 'variable_'
    const StringRef varName = "variable_";
    if (attr.first.starts_with(varName)) {
      unsigned index;
      if (attr.first.drop_front(varName.size()).getAsInteger(10, index))
        return emitError(group.loc, "invalid variable index in template");
      if (index == 0)
        return emitError(group.loc, "variable index must be > 0");
      if (vars.size() < index)
        vars.resize(index);
      if (auto strAttr = dyn_cast<StringAttr>(attr.second))
        vars[index - 1] = strAttr.getValue().str();
    }
  }
  templates[templateName.str()] = vars;
  return success();
}

Attribute LibertyParser::lowerTimingGroup(const LibertyGroup &group) {
  SmallVector<NamedAttribute> attrs;
  for (const auto &attr : group.attrs)
    attrs.push_back(builder.getNamedAttr(attr.first, attr.second));

  llvm::StringMap<SmallVector<Attribute>> subGroups;
  for (const auto &sub : group.subGroups) {
    // Template resolution
    SmallVector<std::string> templateVars;
    bool isTemplate = false;
    if (!sub->args.empty()) {
      if (auto templateName = dyn_cast<StringAttr>(sub->args[0])) {
        auto it = templates.find(templateName.getValue());
        if (it != templates.end()) {
          templateVars = it->second;
          isTemplate = true;
        }
      }
    }

    // Re-implement subAttrs collection using a map to handle duplicates (like
    // 'vector')
    llvm::StringMap<SmallVector<Attribute>> subAttrMap;
    // Add regular attributes
    SmallVector<NamedAttribute> subAttrs;
    llvm::StringMap<SmallVector<Attribute>> accumulatedVectorGroups;

    if (isTemplate) {
      subAttrs.push_back(builder.getNamedAttr("template_name", sub->args[0]));
      SmallVector<Attribute> schema;
      for (const auto &var : templateVars)
        schema.push_back(builder.getStringAttr(var));
      subAttrs.push_back(builder.getNamedAttr("template_schema",
                                              builder.getArrayAttr(schema)));
    } else if (!sub->args.empty()) {
      subAttrs.push_back(
          builder.getNamedAttr("args", builder.getArrayAttr(sub->args)));
    }

    for (const auto &attr : sub->attrs) {
      StringRef attrName = attr.first;
      if (attrName.starts_with("output_current_riseindex_")) {
        unsigned index;
        if (!attrName.drop_front(6).getAsInteger(10, index) && index > 0 &&
            index <= templateVars.size()) {
          attrName = templateVars[index - 1];
        }
      }
      subAttrs.push_back(builder.getNamedAttr(attrName, attr.second));
    }

    for (const auto &child : sub->subGroups) {
      StringRef childName = child->name;

      bool isKnown = false;
      if (childName == "vector" || childName == "values" ||
          childName.starts_with("index_")) {
        isKnown = true;
      } else {
        for (const auto &var : templateVars) {
          if (childName == var) {
            isKnown = true;
            break;
          }
        }
      }

      if (!isKnown && !child->args.empty()) {
        emitWarning(child->loc, "unknown timing subgroup with arguments: ")
            << childName;
      }

      if (childName == "vector") {
        SmallVector<std::string> vectorTemplateVars = templateVars;
        bool vectorIsTemplate = isTemplate;

        if (!vectorIsTemplate && !child->args.empty()) {
          if (auto templateName = dyn_cast<StringAttr>(child->args[0])) {
            auto it = templates.find(templateName.getValue());
            if (it != templates.end()) {
              vectorTemplateVars = it->second;
              vectorIsTemplate = true;
            }
          }
        }

        SmallVector<NamedAttribute> vectorAttrs;
        if (vectorIsTemplate) {
          if (!isTemplate && !child->args.empty()) {
            vectorAttrs.push_back(
                builder.getNamedAttr("template_name", child->args[0]));
            SmallVector<Attribute> schema;
            for (const auto &var : vectorTemplateVars)
              schema.push_back(builder.getStringAttr(var));
            vectorAttrs.push_back(builder.getNamedAttr(
                "template_schema", builder.getArrayAttr(schema)));
          }
        } else if (!child->args.empty()) {
          vectorAttrs.push_back(
              builder.getNamedAttr("args", builder.getArrayAttr(child->args)));
        }

        for (const auto &attr : child->attrs) {
          StringRef attrName = attr.first;
          if (attrName.starts_with("index_")) {
            unsigned index;
            if (!attrName.drop_front(6).getAsInteger(10, index) && index > 0 &&
                index <= vectorTemplateVars.size()) {
              attrName = vectorTemplateVars[index - 1];
            }
          }
          vectorAttrs.push_back(builder.getNamedAttr(attrName, attr.second));
        }

        for (const auto &grandChild : child->subGroups) {
          StringRef gcName = grandChild->name;
          if (gcName.starts_with("index_")) {
            unsigned index;
            if (!gcName.drop_front(6).getAsInteger(10, index) && index > 0 &&
                index <= vectorTemplateVars.size()) {
              gcName = vectorTemplateVars[index - 1];
            }
          }

          bool shouldUnwrap = false;
          if (gcName == "values" || gcName.starts_with("index_")) {
            shouldUnwrap = true;
          } else {
            for (const auto &var : vectorTemplateVars) {
              if (gcName == var) {
                shouldUnwrap = true;
                break;
              }
            }
          }

          Attribute gcAttr;
          if (shouldUnwrap && grandChild->attrs.empty() &&
              grandChild->subGroups.empty() && !grandChild->args.empty()) {
            if (grandChild->args.size() == 1)
              gcAttr = grandChild->args[0];
            else
              gcAttr = builder.getArrayAttr(grandChild->args);
          } else {
            gcAttr = convertGroupToAttr(*grandChild);
          }
          vectorAttrs.push_back(builder.getNamedAttr(gcName, gcAttr));
        }
        accumulatedVectorGroups["vector"].push_back(
            builder.getDictionaryAttr(vectorAttrs));
        continue;
      }

      if (childName.starts_with("index_")) {
        unsigned index;
        if (!childName.drop_front(6).getAsInteger(10, index) && index > 0 &&
            index <= templateVars.size()) {
          childName = templateVars[index - 1];
        }
      }

      bool shouldUnwrap = false;
      if (childName == "values" || childName.starts_with("index_")) {
        shouldUnwrap = true;
      } else {
        for (const auto &var : templateVars) {
          if (childName == var) {
            shouldUnwrap = true;
            break;
          }
        }
      }

      Attribute childAttr;
      if (shouldUnwrap && child->attrs.empty() && child->subGroups.empty() &&
          !child->args.empty()) {
        if (child->args.size() == 1)
          childAttr = child->args[0];
        else
          childAttr = builder.getArrayAttr(child->args);
      } else {
        childAttr = convertGroupToAttr(*child);
      }
      subAttrs.push_back(builder.getNamedAttr(childName, childAttr));
    }

    for (auto &it : accumulatedVectorGroups) {
      subAttrs.push_back(builder.getNamedAttr(
          it.getKey(), builder.getArrayAttr(it.getValue())));
    }

    llvm::DenseSet<StringAttr> seenSubAttrNames;
    for (auto &attr : subAttrs) {
      if (!seenSubAttrNames.insert(attr.getName()).second) {
        emitWarning(lexer.getCurrentLoc(),
                    "duplicate timing subgroup attribute: ")
            << attr.getName();
      }
    }

    subGroups[sub->name].push_back(builder.getDictionaryAttr(subAttrs));
  }

  for (auto &it : subGroups) {
    attrs.push_back(
        builder.getNamedAttr(it.getKey(), builder.getArrayAttr(it.getValue())));
  }

  llvm::DenseSet<StringAttr> seenNames;
  for (auto &attr : attrs) {
    if (!seenNames.insert(attr.getName()).second) {
      emitWarning(lexer.getCurrentLoc(), "duplicate timing group attribute: ")
          << attr.getName();
    }
  }

  return builder.getDictionaryAttr(attrs);
}

ParseResult LibertyParser::lowerCell(const LibertyGroup &group) {
  if (group.args.empty())
    return emitError(lexer.getCurrentLoc(), "cell missing name");

  StringRef cellName;
  if (auto strAttr = dyn_cast<StringAttr>(group.args[0]))
    cellName = strAttr.getValue();
  else
    return emitError(lexer.getCurrentLoc(), "cell name must be a string");

  SmallVector<hw::PortInfo> ports;
  SmallVector<const LibertyGroup *> pinGroups;

  // First pass: gather ports
  for (const auto &sub : group.subGroups) {
    if (sub->name == "pin") {
      pinGroups.push_back(sub.get());
      if (sub->args.empty())
        return emitError(lexer.getCurrentLoc(), "pin missing name");

      StringRef pinName;
      if (auto strAttr = dyn_cast<StringAttr>(sub->args[0]))
        pinName = strAttr.getValue();
      else
        return emitError(lexer.getCurrentLoc(), "pin name must be a string");

      bool isInput = false;
      bool isOutput = false;
      SmallVector<NamedAttribute> pinAttrs;
      for (const auto &attr : sub->attrs) {
        if (attr.first == "direction") {
          if (auto val = dyn_cast<StringAttr>(attr.second)) {
            if (val.getValue() == "input")
              isInput = true;
            else if (val.getValue() == "output")
              isOutput = true;
            else if (val.getValue() == "inout") {
              isInput = true;
              isOutput = true;
            }
          }
          continue;
        }
        pinAttrs.push_back(builder.getNamedAttr(attr.first, attr.second));
      }

      llvm::StringMap<SmallVector<Attribute>> subGroups;
      for (const auto &child : sub->subGroups) {
        // Known timing subgroups.
        if (child->name == "timing" || child->name == "internal_power" ||
            child->name == "output_current_rise" ||
            child->name == "output_current_fall" ||
            child->name == "receiver_capacitance")
          subGroups[child->name].push_back(lowerTimingGroup(*child));
        else
          subGroups[child->name].push_back(convertGroupToAttr(*child));
      }

      for (auto &it : subGroups) {
        pinAttrs.push_back(builder.getNamedAttr(
            it.getKey(), builder.getArrayAttr(it.getValue())));
      }

      auto libertyAttrs = builder.getDictionaryAttr(pinAttrs);
      auto attrs = builder.getDictionaryAttr(
          builder.getNamedAttr("synth.liberty.pin", libertyAttrs));

      if (isInput) {
        hw::PortInfo port;
        port.name = builder.getStringAttr(pinName);
        port.type = builder.getI1Type();
        port.dir = hw::ModulePort::Direction::Input;
        port.attrs = attrs;
        ports.push_back(port);
      }
      if (isOutput) {
        hw::PortInfo port;
        port.name = builder.getStringAttr(pinName);
        port.type = builder.getI1Type();
        port.dir = hw::ModulePort::Direction::Output;
        port.attrs = attrs;
        ports.push_back(port);
      }
    }
  }

  // Fix up argNum for inputs
  int inputIdx = 0;
  for (auto &p : ports) {
    if (p.dir == hw::ModulePort::Direction::Input)
      p.argNum = inputIdx++;
    else
      p.argNum = 0;
  }

  auto loc = builder.getUnknownLoc();
  auto moduleOp = builder.create<hw::HWModuleOp>(
      loc, builder.getStringAttr(cellName), ports);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBodyBlock());

  DenseMap<StringRef, Value> portValues;
  for (size_t i = 0; i < ports.size(); ++i) {
    if (ports[i].dir == hw::ModulePort::Direction::Input)
      portValues[ports[i].name.getValue()] =
          moduleOp.getBodyBlock()->getArgument(ports[i].argNum);
  }

  SmallVector<Value> outputs;
  for (const auto &port : ports) {
    if (port.dir == hw::ModulePort::Direction::Output) {
      const LibertyGroup *pg = nullptr;
      for (auto *g : pinGroups) {
        if (cast<StringAttr>(g->args[0]).getValue() == port.name.getValue()) {
          pg = g;
          break;
        }
      }

      Value val = nullptr;
      if (pg) {
        for (const auto &attr : pg->attrs) {
          if (attr.first == "function") {
            val = parseExpression(cast<StringAttr>(attr.second).getValue(),
                                  portValues);
            break;
          }
        }
      }

      if (!val)
        val = builder.create<hw::ConstantOp>(loc, builder.getI1Type(), 0);
      outputs.push_back(val);
    }
  }

  auto *block = moduleOp.getBodyBlock();
  block->getTerminator()->setOperands(outputs);
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

  // Attribute statement.
  if (lexer.peekToken().kind == LibertyTokenKind::Colon) {
    lexer.nextToken(); // :
    Attribute val;
    if (parseAttribute(val))
      return failure();
    parent.attrs.emplace_back(name, val);
    return consume(LibertyTokenKind::Semi, "expected ';'");
  }

  // Group statement.
  if (lexer.peekToken().kind == LibertyTokenKind::LParen) {
    auto subGroup = std::make_unique<LibertyGroup>();
    subGroup->name = name;
    subGroup->loc = nameTok.location;
    if (parseGroupBody(*subGroup))
      return failure();
    parent.subGroups.push_back(std::move(subGroup));
    return success();
  }

  // TODO: Support define.

  return emitError(nameTok.location, "expected ':' or '('");
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
    StringRef str = getTokenSpelling(token);

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

} // namespace

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
