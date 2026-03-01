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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

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
///   UnaryExpr -> '!' UnaryExpr | '(' OrExpr ')' ['\''] | ID ['\']
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

  ParseResult parseOrExpr(Value &result);
  ParseResult parseXorExpr(Value &result);
  ParseResult parseAndExpr(Value &result);
  ParseResult parseUnaryExpr(Value &result);

  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message) const;
  InFlightDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message) const;
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
    trueVal = hw::ConstantOp::create(builder, lexer.translateLocation(loc),
                                     builder.getI1Type(), 1);
  return comb::XorOp::create(builder, lexer.translateLocation(loc), val,
                             trueVal);
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

InFlightDiagnostic ExpressionParser::emitWarning(llvm::SMLoc loc,
                                                 const Twine &message) const {
  return mlir::emitWarning(lexer.translateLocation(loc), message);
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

static std::optional<double> parseTimeUnitToPs(StringRef unit) {
  unit = unit.trim();
  if (unit.empty())
    return std::nullopt;

  size_t split = 0;
  while (split < unit.size()) {
    char c = unit[split];
    if (llvm::isDigit(c) || c == '+' || c == '-' || c == '.' || c == 'e' ||
        c == 'E') {
      ++split;
      continue;
    }
    break;
  }
  if (split == 0)
    return std::nullopt;

  double magnitude = 0.0;
  if (unit.take_front(split).getAsDouble(magnitude))
    return std::nullopt;

  llvm::SmallString<8> suffixStorage = unit.drop_front(split).trim();
  for (char &c : suffixStorage)
    c = llvm::toLower(c);
  StringRef suffix = suffixStorage;

  double unitToPs = 0.0;
  if (suffix == "s")
    unitToPs = 1.0e12;
  else if (suffix == "ms")
    unitToPs = 1.0e9;
  else if (suffix == "us")
    unitToPs = 1.0e6;
  else if (suffix == "ns")
    unitToPs = 1.0e3;
  else if (suffix == "ps")
    unitToPs = 1.0;
  else if (suffix == "fs")
    unitToPs = 1.0e-3;
  else
    return std::nullopt;

  return magnitude * unitToPs;
}

static std::optional<ArrayAttr> parseValueList(Attribute attr,
                                               OpBuilder &builder) {
  if (!attr)
    return std::nullopt;

  SmallVector<Attribute> values;

  auto addValue = [&](double value) {
    values.push_back(builder.getF64FloatAttr(value));
  };

  if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    addValue(floatAttr.getValueAsDouble());
    return builder.getArrayAttr(values);
  }
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    addValue(static_cast<double>(intAttr.getInt()));
    return builder.getArrayAttr(values);
  }
  if (auto arr = dyn_cast<ArrayAttr>(attr)) {
    for (auto item : arr) {
      if (auto f = dyn_cast<FloatAttr>(item))
        addValue(f.getValueAsDouble());
      else if (auto i = dyn_cast<IntegerAttr>(item))
        addValue(static_cast<double>(i.getInt()));
      else
        return std::nullopt;
    }
    return builder.getArrayAttr(values);
  }

  auto strAttr = dyn_cast<StringAttr>(attr);
  if (!strAttr)
    return std::nullopt;

  SmallVector<StringRef> parts;
  strAttr.getValue().split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef part : parts) {
    double value = 0.0;
    if (part.trim().getAsDouble(value))
      return std::nullopt;
    addValue(value);
  }
  if (values.empty())
    return std::nullopt;
  return builder.getArrayAttr(values);
}

static std::optional<ArrayAttr> getGroupField(const LibertyGroup &group,
                                              StringRef fieldName,
                                              OpBuilder &builder) {
  if (auto value = group.getAttribute(fieldName).first)
    if (auto parsed = parseValueList(value, builder))
      return parsed;

  for (const auto &sub : group.subGroups) {
    if (sub->name != fieldName || sub->args.empty())
      continue;
    if (auto parsed = parseValueList(sub->args.front(), builder))
      return parsed;
  }
  return std::nullopt;
}

static std::optional<ArrayAttr> getFirstTableField(const LibertyGroup &timing,
                                                   StringRef tableName,
                                                   StringRef fieldName,
                                                   OpBuilder &builder) {
  for (const auto &sub : timing.subGroups) {
    if (sub->name != tableName)
      continue;
    if (auto values = getGroupField(*sub, fieldName, builder))
      return values;
  }
  return std::nullopt;
}

static std::optional<double> getFirstNumeric(ArrayAttr values) {
  if (values.empty())
    return std::nullopt;
  auto first = values[0];
  if (auto f = dyn_cast<FloatAttr>(first))
    return f.getValueAsDouble();
  if (auto i = dyn_cast<IntegerAttr>(first))
    return static_cast<double>(i.getInt());
  return std::nullopt;
}

struct CcsVectorTableData {
  ArrayAttr selectorIndex1;
  ArrayAttr selectorIndex2;
  ArrayAttr referenceTimes;
  ArrayAttr vectorTimes;
  ArrayAttr vectorValues;
};

static CcsVectorTableData extractCcsVectorTable(const LibertyGroup &timing,
                                                StringRef tableName,
                                                OpBuilder &builder) {
  SmallVector<Attribute> selector1;
  SmallVector<Attribute> selector2;
  SmallVector<Attribute> refs;
  SmallVector<Attribute> times;
  SmallVector<Attribute> values;

  for (const auto &table : timing.subGroups) {
    if (table->name != tableName)
      continue;
    for (const auto &vector : table->subGroups) {
      if (vector->name != "vector")
        continue;
      auto vectorTimes = getGroupField(*vector, "index_3", builder);
      auto vectorValues = getGroupField(*vector, "values", builder);
      if (!vectorTimes || !vectorValues)
        continue;

      auto index1 = getGroupField(*vector, "index_1", builder);
      auto index2 = getGroupField(*vector, "index_2", builder);
      auto referenceTime = getGroupField(*vector, "reference_time", builder);

      selector1.push_back(builder.getF64FloatAttr(
          getFirstNumeric(index1.value_or(builder.getArrayAttr({})))
              .value_or(0.0)));
      selector2.push_back(builder.getF64FloatAttr(
          getFirstNumeric(index2.value_or(builder.getArrayAttr({})))
              .value_or(0.0)));
      refs.push_back(builder.getF64FloatAttr(
          getFirstNumeric(referenceTime.value_or(builder.getArrayAttr({})))
              .value_or(0.0)));
      times.push_back(*vectorTimes);
      values.push_back(*vectorValues);
    }
  }

  CcsVectorTableData result;
  result.selectorIndex1 = builder.getArrayAttr(selector1);
  result.selectorIndex2 = builder.getArrayAttr(selector2);
  result.referenceTimes = builder.getArrayAttr(refs);
  result.vectorTimes = builder.getArrayAttr(times);
  result.vectorValues = builder.getArrayAttr(values);
  return result;
}

static std::optional<circt::synth::NLDMArcAttr>
buildNldmArcAttr(const LibertyGroup &timing, StringRef outputPin,
                 OpBuilder &builder) {
  StringAttr relatedPin;
  auto relatedAttr = timing.getAttribute("related_pin").first;
  if (auto str = dyn_cast<StringAttr>(relatedAttr))
    relatedPin = str;
  else if (!timing.args.empty())
    relatedPin = dyn_cast<StringAttr>(timing.args.front());

  if (!relatedPin)
    return std::nullopt;

  auto toPin = builder.getStringAttr(outputPin);
  auto timingSense =
      dyn_cast<StringAttr>(timing.getAttribute("timing_sense").first);
  if (!timingSense)
    timingSense = builder.getStringAttr("");

  auto riseIndex1 = getFirstTableField(timing, "cell_rise", "index_1", builder)
                        .value_or(builder.getArrayAttr({}));
  auto riseIndex2 = getFirstTableField(timing, "cell_rise", "index_2", builder)
                        .value_or(builder.getArrayAttr({}));
  auto riseValues = getFirstTableField(timing, "cell_rise", "values", builder)
                        .value_or(builder.getArrayAttr({}));

  auto fallIndex1 = getFirstTableField(timing, "cell_fall", "index_1", builder)
                        .value_or(builder.getArrayAttr({}));
  auto fallIndex2 = getFirstTableField(timing, "cell_fall", "index_2", builder)
                        .value_or(builder.getArrayAttr({}));
  auto fallValues = getFirstTableField(timing, "cell_fall", "values", builder)
                        .value_or(builder.getArrayAttr({}));

  auto riseTransitionIndex1 =
      getFirstTableField(timing, "rise_transition", "index_1", builder)
          .value_or(builder.getArrayAttr({}));
  auto riseTransitionIndex2 =
      getFirstTableField(timing, "rise_transition", "index_2", builder)
          .value_or(builder.getArrayAttr({}));
  auto riseTransitionValues =
      getFirstTableField(timing, "rise_transition", "values", builder)
          .value_or(builder.getArrayAttr({}));

  auto fallTransitionIndex1 =
      getFirstTableField(timing, "fall_transition", "index_1", builder)
          .value_or(builder.getArrayAttr({}));
  auto fallTransitionIndex2 =
      getFirstTableField(timing, "fall_transition", "index_2", builder)
          .value_or(builder.getArrayAttr({}));
  auto fallTransitionValues =
      getFirstTableField(timing, "fall_transition", "values", builder)
          .value_or(builder.getArrayAttr({}));

  return circt::synth::NLDMArcAttr::get(
      builder.getContext(), relatedPin, toPin, timingSense, riseIndex1,
      riseIndex2, riseValues, fallIndex1, fallIndex2, fallValues,
      riseTransitionIndex1, riseTransitionIndex2, riseTransitionValues,
      fallTransitionIndex1, fallTransitionIndex2, fallTransitionValues);
}

static std::optional<circt::synth::CCSPilotArcAttr>
buildCcsPilotArcAttr(const LibertyGroup &timing, StringRef outputPin,
                     OpBuilder &builder) {
  StringAttr relatedPin;
  auto relatedAttr = timing.getAttribute("related_pin").first;
  if (auto str = dyn_cast<StringAttr>(relatedAttr))
    relatedPin = str;
  else if (!timing.args.empty())
    relatedPin = dyn_cast<StringAttr>(timing.args.front());
  if (!relatedPin)
    return std::nullopt;

  auto toPin = builder.getStringAttr(outputPin);

  auto riseTimes =
      getFirstTableField(timing, "output_current_rise", "index_1", builder)
          .value_or(builder.getArrayAttr({}));
  auto riseValues =
      getFirstTableField(timing, "output_current_rise", "values", builder)
          .value_or(builder.getArrayAttr({}));
  auto fallTimes =
      getFirstTableField(timing, "output_current_fall", "index_1", builder)
          .value_or(builder.getArrayAttr({}));
  auto fallValues =
      getFirstTableField(timing, "output_current_fall", "values", builder)
          .value_or(builder.getArrayAttr({}));

  auto riseVectors =
      extractCcsVectorTable(timing, "output_current_rise", builder);
  auto fallVectors =
      extractCcsVectorTable(timing, "output_current_fall", builder);

  if (riseTimes.empty() && riseValues.empty() &&
      !riseVectors.vectorTimes.empty() && !riseVectors.vectorValues.empty()) {
    if (auto firstTimes = dyn_cast<ArrayAttr>(riseVectors.vectorTimes[0]))
      riseTimes = firstTimes;
    if (auto firstValues = dyn_cast<ArrayAttr>(riseVectors.vectorValues[0]))
      riseValues = firstValues;
  }
  if (fallTimes.empty() && fallValues.empty() &&
      !fallVectors.vectorTimes.empty() && !fallVectors.vectorValues.empty()) {
    if (auto firstTimes = dyn_cast<ArrayAttr>(fallVectors.vectorTimes[0]))
      fallTimes = firstTimes;
    if (auto firstValues = dyn_cast<ArrayAttr>(fallVectors.vectorValues[0]))
      fallValues = firstValues;
  }

  if (riseTimes.empty() && riseValues.empty() && fallTimes.empty() &&
      fallValues.empty() && riseVectors.vectorTimes.empty() &&
      fallVectors.vectorTimes.empty())
    return std::nullopt;

  return circt::synth::CCSPilotArcAttr::get(
      builder.getContext(), relatedPin, toPin, riseTimes, riseValues, fallTimes,
      fallValues, riseVectors.selectorIndex1, riseVectors.selectorIndex2,
      riseVectors.referenceTimes, riseVectors.vectorTimes,
      riseVectors.vectorValues, fallVectors.selectorIndex1,
      fallVectors.selectorIndex2, fallVectors.referenceTimes,
      fallVectors.vectorTimes, fallVectors.vectorValues);
}

static std::optional<circt::synth::CCSPilotReceiverAttr>
buildCcsPilotReceiverAttr(const LibertyGroup &timing, StringRef outputPin,
                          OpBuilder &builder) {
  StringAttr relatedPin;
  auto relatedAttr = timing.getAttribute("related_pin").first;
  if (auto str = dyn_cast<StringAttr>(relatedAttr))
    relatedPin = str;
  else if (!timing.args.empty())
    relatedPin = dyn_cast<StringAttr>(timing.args.front());
  if (!relatedPin)
    return std::nullopt;

  auto toPin = builder.getStringAttr(outputPin);

  auto cap1RiseIndex1 = getFirstTableField(timing, "receiver_capacitance1_rise",
                                           "index_1", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap1RiseIndex2 = getFirstTableField(timing, "receiver_capacitance1_rise",
                                           "index_2", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap1RiseValues = getFirstTableField(timing, "receiver_capacitance1_rise",
                                           "values", builder)
                            .value_or(builder.getArrayAttr({}));

  auto cap2RiseIndex1 = getFirstTableField(timing, "receiver_capacitance2_rise",
                                           "index_1", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap2RiseIndex2 = getFirstTableField(timing, "receiver_capacitance2_rise",
                                           "index_2", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap2RiseValues = getFirstTableField(timing, "receiver_capacitance2_rise",
                                           "values", builder)
                            .value_or(builder.getArrayAttr({}));

  auto cap1FallIndex1 = getFirstTableField(timing, "receiver_capacitance1_fall",
                                           "index_1", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap1FallIndex2 = getFirstTableField(timing, "receiver_capacitance1_fall",
                                           "index_2", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap1FallValues = getFirstTableField(timing, "receiver_capacitance1_fall",
                                           "values", builder)
                            .value_or(builder.getArrayAttr({}));

  auto cap2FallIndex1 = getFirstTableField(timing, "receiver_capacitance2_fall",
                                           "index_1", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap2FallIndex2 = getFirstTableField(timing, "receiver_capacitance2_fall",
                                           "index_2", builder)
                            .value_or(builder.getArrayAttr({}));
  auto cap2FallValues = getFirstTableField(timing, "receiver_capacitance2_fall",
                                           "values", builder)
                            .value_or(builder.getArrayAttr({}));

  if (cap1RiseValues.empty() && cap2RiseValues.empty() &&
      cap1FallValues.empty() && cap2FallValues.empty())
    return std::nullopt;

  return circt::synth::CCSPilotReceiverAttr::get(
      builder.getContext(), relatedPin, toPin, cap1RiseIndex1, cap1RiseIndex2,
      cap1RiseValues, cap2RiseIndex1, cap2RiseIndex2, cap2RiseValues,
      cap1FallIndex1, cap1FallIndex2, cap1FallValues, cap2FallIndex1,
      cap2FallIndex2, cap2FallValues);
}

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
  ParseResult parseLibrary();
  ParseResult parseGroupBody(LibertyGroup &group);
  ParseResult parseStatement(LibertyGroup &parent);

  // Lowering methods
  ParseResult lowerCell(const LibertyGroup &group, StringAttr &cellNameAttr);

  //===--------------------------------------------------------------------===//
  // Parser for Subgroup of "cell" group.
  //===--------------------------------------------------------------------===//

  Attribute convertGroupToAttr(const LibertyGroup &group);

  // Attribute parsing
  ParseResult parseAttribute(Attribute &result);
  static ParseResult parseAttribute(Attribute &result, OpBuilder &builder,
                                    StringRef attr);

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
    } else if (*curPtr == '.') {
      if (seenDot) {
        mlir::emitError(translateLocation(SMLoc::getFromPointer(curPtr)),
                        "multiple decimal points in number");
        return makeToken(LibertyTokenKind::Error, start);
      }
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
    if (token.kind != LibertyTokenKind::Identifier ||
        token.spelling != "library") {
      return emitError(token.location, "expected `library` keyword");
    }
    lexer.nextToken();
    if (parseLibrary())
      return failure();
  }
  return success();
}

ParseResult LibertyParser::parseLibrary() {
  LibertyGroup libertyLib;
  if (parseGroupBody(libertyLib))
    return failure();

  DenseSet<StringAttr> seenCells;
  for (auto &stmt : libertyLib.subGroups) {
    // TODO: Support more group types
    if (stmt->name == "cell") {
      StringAttr cellNameAttr;
      if (lowerCell(*stmt, cellNameAttr))
        return failure();
      if (!seenCells.insert(cellNameAttr).second)
        return emitError(stmt->loc, "redefinition of cell '" +
                                        cellNameAttr.getValue() + "'");
      continue;
    }
  }

  libertyLib.eraseSubGroup(
      [](const LibertyGroup &group) { return group.name == "cell"; });
  auto attr = convertGroupToAttr(libertyLib);
  module->setAttr("synth.liberty.library", attr);

  if (auto timeUnitAttr =
          dyn_cast<StringAttr>(libertyLib.getAttribute("time_unit").first)) {
    if (auto ps = parseTimeUnitToPs(timeUnitAttr.getValue()))
      module->setAttr("synth.nldm.time_unit",
                      circt::synth::NLDMTimeUnitAttr::get(
                          builder.getContext(), builder.getF64FloatAttr(*ps)));
  }

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

ParseResult LibertyParser::lowerCell(const LibertyGroup &group,
                                     StringAttr &cellNameAttr) {
  if (group.args.empty())
    return emitError(group.loc, "cell missing name");

  cellNameAttr = dyn_cast<StringAttr>(group.args[0]);
  if (!cellNameAttr)
    return emitError(group.loc, "cell name must be a string");

  SmallVector<hw::PortInfo> ports;
  SmallVector<const LibertyGroup *> pinGroups;
  llvm::DenseSet<StringAttr> seenPins;

  // First pass: gather ports
  for (const auto &sub : group.subGroups) {
    if (sub->name != "pin")
      continue;

    pinGroups.push_back(sub.get());
    if (sub->args.empty())
      return emitError(sub->loc, "pin missing name");

    StringAttr pinName = dyn_cast<StringAttr>(sub->args[0]);
    if (!seenPins.insert(pinName).second)
      return emitError(sub->loc,
                       "redefinition of pin '" + pinName.getValue() + "'");

    std::optional<hw::ModulePort::Direction> dir;
    SmallVector<NamedAttribute> pinAttrs;
    SmallVector<Attribute> nldmArcs;
    SmallVector<Attribute> ccsPilotArcs;
    SmallVector<Attribute> ccsPilotReceivers;

    for (const auto &attr : sub->attrs) {
      if (attr.name == "direction") {
        if (auto val = dyn_cast<StringAttr>(attr.value)) {
          if (val.getValue() == "input")
            dir = hw::ModulePort::Direction::Input;
          else if (val.getValue() == "output")
            dir = hw::ModulePort::Direction::Output;
          else if (val.getValue() == "inout")
            dir = hw::ModulePort::Direction::InOut;
          else
            return emitError(sub->loc,
                             "pin direction must be input, output, or inout");
        } else {
          return emitError(sub->loc,
                           "pin direction must be a string attribute");
        }
        continue;
      }
      pinAttrs.push_back(builder.getNamedAttr(attr.name, attr.value));
    }

    if (!dir)
      return emitError(sub->loc, "pin direction must be specified");

    llvm::StringMap<SmallVector<Attribute>> subGroups;
    for (const auto &child : sub->subGroups) {
      if (child->name == "timing") {
        if (auto nldmArc =
                buildNldmArcAttr(*child, pinName.getValue(), builder))
          nldmArcs.push_back(*nldmArc);
        if (auto ccsArc =
                buildCcsPilotArcAttr(*child, pinName.getValue(), builder))
          ccsPilotArcs.push_back(*ccsArc);
        if (auto ccsReceiver =
                buildCcsPilotReceiverAttr(*child, pinName.getValue(), builder))
          ccsPilotReceivers.push_back(*ccsReceiver);
      }

      subGroups[child->name].push_back(convertGroupToAttr(*child));
    }

    for (auto &it : subGroups)
      pinAttrs.push_back(builder.getNamedAttr(
          it.getKey(), builder.getArrayAttr(it.getValue())));

    if (!nldmArcs.empty())
      pinAttrs.push_back(builder.getNamedAttr("synth.nldm.arcs",
                                              builder.getArrayAttr(nldmArcs)));
    if (!ccsPilotArcs.empty())
      pinAttrs.push_back(builder.getNamedAttr(
          "synth.ccs.pilot.arcs", builder.getArrayAttr(ccsPilotArcs)));
    if (!ccsPilotReceivers.empty())
      pinAttrs.push_back(
          builder.getNamedAttr("synth.ccs.pilot.receivers",
                               builder.getArrayAttr(ccsPilotReceivers)));

    auto libertyAttrs = builder.getDictionaryAttr(pinAttrs);
    auto attrs = builder.getDictionaryAttr(
        builder.getNamedAttr("synth.liberty.pin", libertyAttrs));

    hw::PortInfo port;
    port.name = pinName;
    port.type = builder.getI1Type();
    port.dir = *dir;
    port.attrs = attrs;
    ports.push_back(port);
  }

  // Fix up argNum for inputs
  int inputIdx = 0;
  for (auto &p : ports) {
    if (p.dir == hw::ModulePort::Direction::Input)
      p.argNum = inputIdx++;
    else
      p.argNum = 0;
  }

  auto loc = lexer.translateLocation(group.loc);
  auto moduleOp = hw::HWModuleOp::create(builder, loc, cellNameAttr, ports);

  // Emit cell area as a module attribute for downstream passes.
  if (auto areaAttr = group.getAttribute("area").first)
    if (auto areaFloat = dyn_cast<FloatAttr>(areaAttr))
      moduleOp->setAttr("synth.liberty.area", areaFloat);

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
    if (!attrPair.first)
      return emitError(pg->loc, "expected function attribute");
    Value val;
    if (parseExpression(val, cast<StringAttr>(attrPair.first).getValue(),
                        portValues, attrPair.second))
      return failure();
    outputs.push_back(val);
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
    auto loc = lexer.peekToken().location;
    if (parseAttribute(val))
      return failure();
    parent.attrs.emplace_back(name, val, loc);
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

ParseResult LibertyParser::parseAttribute(Attribute &result, OpBuilder &builder,
                                          StringRef attr) {
  double val;
  if (!attr.getAsDouble(val)) {
    result = builder.getF64FloatAttr(val);
  } else {
    // Keep as string if not a valid number
    result = builder.getStringAttr(attr);
  }
  return success();
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
    double val;
    if (!numStr.getAsDouble(val)) {
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
    StringRef str = getTokenSpelling(token);
    result = builder.getStringAttr(str);
    return success();
  }

  // TODO: Add array for timing attributes
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
