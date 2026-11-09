//===- FIRRTLSourceGen.cpp - FIRRTL source syntax glue ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <memory>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

namespace {
bool isTokenIdentifier(StringRef name) {
  return !name.empty() && (isAlpha(name.front()) || name.front() == '_') &&
         all_of(name.drop_front(),
                [](char c) { return isAlnum(c) || c == '_'; });
}

struct Version {
  bool unspecified = false, next = false;
  int64_t major = 0, minor = 0, patch = 0;

  explicit Version(const Record *def) {
    if (def->getName() == "FIRRTLUnspecifiedVersion") {
      unspecified = true;
      return;
    }
    if (def->getName() == "FIRRTLNextVersion") {
      next = true;
      return;
    }
    if (!def->isSubClassOf("FIRRTLReleasedVersion"))
      PrintFatalError(
          def->getLoc(),
          "expected a released, unspecified, or next FIRRTL source version");
    major = def->getValueAsInt("majorVersion");
    minor = def->getValueAsInt("minorVersion");
    patch = def->getValueAsInt("patchVersion");
    if (major < 0 || minor < 0 || patch < 0 || major > 65535 || minor > 65535 ||
        patch > 65535)
      PrintFatalError(def->getLoc(),
                      "FIRRTL version components must fit in uint16_t");
  }

  bool isBase() const {
    return !unspecified && !next && major == 0 && minor == 0 && patch == 0;
  }
  void emit(raw_ostream &os) const {
    if (unspecified)
      os << "missingSpecFIRVersion";
    else if (next)
      os << "nextFIRVersion";
    else
      os << "FIRVersion(" << major << ", " << minor << ", " << patch << ")";
  }
};

void emitString(raw_ostream &os, StringRef str) {
  os << '"';
  os.write_escaped(str);
  os << '"';
}

struct Primitive {
  Operator op;
  StringRef spelling;
  Version version;
  bool hasCustomParser, transparent = false, usesOperandDirective = false;
  unsigned numOperands = 0;
  SmallVector<std::string> getters;
  std::string operandGetter;

  explicit Primitive(const Record *def)
      : op(def), version(def->getValueAsDef("firrtlVersion")),
        hasCustomParser(def->getValueAsBit("firrtlHasCustomParser")) {
    parseFormat(def);
    if (!transparent && !isTokenIdentifier(spelling))
      PrintFatalError(
          def->getLoc(),
          "primitive spelling must be a lexer and C++ token identifier");
    if (op.getNumResults() != 1 || op.getResult(0).isVariableLength() ||
        op.getNumRegions() || op.getNumSuccessors())
      PrintFatalError(def->getLoc(), "primitive must have exactly one fixed "
                                     "result and no regions or successors");
    for (auto arg : op.getArgs()) {
      if (auto *operand = dyn_cast<NamedTypeConstraint *>(arg)) {
        if (!getters.empty() || operand->isOptional() ||
            (operand->isVariadic() &&
             !(hasCustomParser && usesOperandDirective)) ||
            operand->name.empty())
          PrintFatalError(def->getLoc(), "primitive operands must be fixed and "
                                         "precede integer attributes");
        ++numOperands;
        if (numOperands == 1)
          operandGetter = op.getGetterName(operand->name);
      } else if (auto *attr = dyn_cast<NamedAttribute *>(arg)) {
        if (attr->attr.getDef().getName() != "I32Attr" ||
            attr->attr.isOptional() || attr->attr.hasDefaultValue() ||
            attr->name.empty())
          PrintFatalError(
              def->getLoc(),
              "primitive attributes must be named, required I32Attr arguments");
        getters.push_back(op.getGetterName(attr->name));
      } else {
        PrintFatalError(def->getLoc(),
                        "primitive properties are not supported");
      }
    }
    if (numOperands == 0)
      PrintFatalError(def->getLoc(),
                      "primitive must have an expression operand");
    if (transparent && (numOperands != 1 || !getters.empty()))
      PrintFatalError(
          def->getLoc(),
          "transparent format must have one fixed operand and no attributes");
  }

  void parseFormat(const Record *def) {
    StringRef remaining = def->getValueAsString("firrtlFormat").trim();
    if (remaining.starts_with("$")) {
      // A bare operand cannot reconstruct the operation during import.
      if (hasCustomParser)
        PrintFatalError(def->getLoc(),
                        "transparent format cannot have a custom parser");
      if (op.getNumArgs() != 1)
        PrintFatalError(
            def->getLoc(),
            "transparent format must have one fixed operand and no attributes");
      auto *operand = dyn_cast<NamedTypeConstraint *>(op.getArg(0));
      if (!operand || operand->isVariableLength() || operand->name.empty() ||
          !remaining.consume_front("$") ||
          !remaining.consume_front(operand->name) || !remaining.trim().empty())
        PrintFatalError(
            def->getLoc(),
            "transparent format must reference its single fixed operand");
      transparent = true;
      return;
    }
    auto fail = [&]() {
      PrintFatalError(def->getLoc(),
                      "invalid firrtlFormat: expected a call with "
                      "all ODS arguments in order");
    };
    auto literal = [&]() -> StringRef {
      remaining = remaining.ltrim();
      if (!remaining.consume_front("`"))
        fail();
      auto end = remaining.find('`');
      if (end == StringRef::npos)
        fail();
      StringRef value = remaining.take_front(end);
      remaining = remaining.drop_front(end + 1);
      return value;
    };
    spelling = literal();
    if (literal() != "(")
      fail();
    // The operands directive represents the entire leading operand group,
    // including its internal commas. Attributes remain explicit.
    int argIndex = 0;
    remaining = remaining.ltrim();
    if (remaining.consume_front("operands")) {
      usesOperandDirective = true;
      while (argIndex < op.getNumArgs() &&
             isa<NamedTypeConstraint *>(op.getArg(argIndex)))
        ++argIndex;
      if (argIndex == 0)
        fail();
    }
    for (; argIndex < op.getNumArgs(); ++argIndex) {
      if (argIndex != 0 && literal() != ",")
        fail();
      auto arg = op.getArg(argIndex);
      StringRef name;
      if (auto *operand = dyn_cast<NamedTypeConstraint *>(arg))
        name = operand->name;
      else if (auto *attr = dyn_cast<NamedAttribute *>(arg))
        name = attr->name;
      else
        PrintFatalError(def->getLoc(),
                        "primitive properties are not supported");
      remaining = remaining.ltrim();
      if (name.empty() || !remaining.consume_front("$") ||
          !remaining.consume_front(name) ||
          (!remaining.empty() &&
           (isAlnum(remaining.front()) || remaining.front() == '_')))
        fail();
    }
    if (literal() != ")" || !remaining.trim().empty())
      fail();
  }
};

} // namespace

static bool emitPrimitives(const RecordKeeper &records, raw_ostream &os) {
  SmallVector<std::unique_ptr<Primitive>> primitives;
  llvm::StringSet<> spellings;
  for (const Record *def : records.getAllDerivedDefinitions("FIRRTLOp")) {
    if (def->getValueAsString("firrtlFormat").empty())
      continue;
    primitives.push_back(std::make_unique<Primitive>(def));
    if (!primitives.back()->transparent &&
        !spellings.insert(primitives.back()->spelling).second)
      PrintFatalError(def->getLoc(), "duplicate FIRRTL primitive spelling '" +
                                         primitives.back()->spelling + "'");
  }
  llvm::sort(primitives, [](const auto &a, const auto &b) {
    return a->op.getDef().getName() < b->op.getDef().getName();
  });

  os << "// Generated by circt-tblgen; do not edit.\n\n";
  os << "#ifdef GET_FIRRTL_PRIMITIVE_TOKENS\n";
  for (const auto &primitive : primitives) {
    const auto &p = *primitive;
    if (p.transparent || p.hasCustomParser)
      continue;
    os << "TOK_LPKEYWORD_PRIM(" << p.spelling << ", " << p.op.getCppClassName()
       << ", " << p.numOperands << ", " << p.getters.size() << ", ";
    p.version.emit(os);
    os << ", ";
    emitString(os, p.spelling);
    os << ")\n";
  }
  os << "#endif\n\n#ifdef GET_FIRRTL_PRIMITIVE_EMITTERS\n";
  for (const auto &primitive : primitives) {
    const auto &p = *primitive;
    os << "void emitExpression(" << p.op.getCppClassName() << " op) {\n";
    if (!p.version.isBase()) {
      os << "  if (failed(requireVersion(";
      p.version.emit(os);
      os << ", op, ";
      emitString(os, p.transparent ? p.op.getDef().getValueAsString("opName")
                                   : p.spelling);
      os << "))) return;\n";
    }
    if (p.transparent) {
      os << "  emitExpression(op." << p.operandGetter << "());\n}\n";
      continue;
    }
    // A fixed unary call has no additional call or operand boxes.
    if (p.hasCustomParser && p.numOperands == 1 && p.getters.empty() &&
        !p.op.getOperand(0).isVariableLength()) {
      os << "  ps << ";
      emitString(os, (p.spelling + "(").str());
      os << ";\n  emitExpression(op." << p.operandGetter
         << "());\n  ps << \")\";\n}\n";
      continue;
    }
    os << "  emitPrimExpr(";
    emitString(os, p.spelling);
    os << ", op";
    if (!p.getters.empty()) {
      os << ", ";
      if (p.getters.size() > 1)
        os << "{";
      interleaveComma(p.getters, os, [&](const std::string &getter) {
        os << "op." << getter << "()";
      });
      if (p.getters.size() > 1)
        os << "}";
    }
    os << ");\n}\n";
  }
  os << "#endif\n\n#ifdef GET_FIRRTL_PRIMITIVE_DISPATCH\n";
  for (const auto &p : primitives)
    os << p->op.getCppClassName() << ",\n";
  os << "#endif\n";
  return false;
}

static GenRegistration genPrimitives("firrtl-primitives",
                                     "Generate FIRRTL source primitive glue",
                                     emitPrimitives);
