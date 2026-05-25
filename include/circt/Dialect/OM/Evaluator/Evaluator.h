//===- Evaluator.h - Object Model dialect evaluator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
#define CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H

#include "circt/Dialect/OM/OMOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"

#include <utility>

namespace circt {
namespace om {

namespace evaluator {
class EvaluatorValue;

/// A value of an object in memory. It is either a composite Object, or a
/// primitive Attribute. Further refinement is expected.
using EvaluatorValuePtr = std::shared_ptr<EvaluatorValue>;

/// The fields of a composite Object, currently represented as a map. Further
/// refinement is expected.
using ObjectFields = SmallDenseMap<StringAttr, EvaluatorValuePtr>;

/// Base class for evaluator runtime values.
/// Enables the shared_from_this functionality so Evaluator Value pointers can
/// be passed through the CAPI and unwrapped back into C++ smart pointers with
/// the appropriate reference count.
class EvaluatorValue : public std::enable_shared_from_this<EvaluatorValue> {
public:
  // Implement LLVM RTTI.
  enum class Kind { Attr, Object, List, BasePath, Path, Unknown };
  EvaluatorValue(MLIRContext *ctx, Kind kind, Location loc)
      : kind(kind), ctx(ctx), loc(loc) {}
  Kind getKind() const { return kind; }
  MLIRContext *getContext() const { return ctx; }

  /// Return true if this value represents an `om.unknown`.
  bool isUnknown() const { return getKind() == Kind::Unknown; }

  /// Return the associated MLIR context.
  MLIRContext *getContext() { return ctx; }

  // Return a MLIR type which the value represents.
  Type getType() const;

  // Return the Location associated with the Value.
  Location getLoc() const { return loc; }
  // Set the Location associated with the Value.
  void setLoc(Location l) { loc = l; }
  // Set the Location, if it is unknown.
  void setLocIfUnknown(Location l) {
    if (isa<UnknownLoc>(loc))
      loc = l;
  }

private:
  const Kind kind;
  MLIRContext *ctx;
  Location loc;
};

/// Values which can be directly representable by MLIR attributes.
class AttributeValue : public EvaluatorValue {
public:
  Attribute getAttr() const { return attr; }
  template <typename AttrTy>
  AttrTy getAs() const {
    return dyn_cast<AttrTy>(attr);
  }
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Attr;
  }

  Type getType() const { return type; }

  // Factory methods that create AttributeValue objects
  static std::shared_ptr<EvaluatorValue> get(Attribute attr,
                                             LocationAttr loc = {});
  static std::shared_ptr<EvaluatorValue> get(Type type, LocationAttr loc = {});

private:
  // Make AttributeValue constructible only by the factory methods
  struct PrivateTag {};

  // Constructor that requires a PrivateTag
  AttributeValue(PrivateTag, Attribute attr, Location loc)
      : EvaluatorValue(attr.getContext(), Kind::Attr, loc), attr(attr),
        type(cast<TypedAttr>(attr).getType()) {}

  // Constructor for unknown AttributeValues whose payload is not materialized.
  AttributeValue(PrivateTag, Type type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::Attr, loc), type(type) {}

  Attribute attr = {};
  Type type;

  // Friend declaration for the factory methods
  friend std::shared_ptr<EvaluatorValue> get(Attribute attr, LocationAttr loc);
  friend std::shared_ptr<EvaluatorValue> get(Type type, LocationAttr loc);
};

class UnknownValue : public EvaluatorValue {
public:
  UnknownValue(Type type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::Unknown, loc), type(type) {}

  Type getType() const { return type; }

  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Unknown;
  }

private:
  Type type;
};

/// A List which contains variadic length of elements with the same type.
class ListValue : public EvaluatorValue {
public:
  ListValue(om::ListType type, SmallVector<EvaluatorValuePtr> elements,
            Location loc)
      : EvaluatorValue(type.getContext(), Kind::List, loc), type(type),
        elements(std::move(elements)), elementsInitialized(true) {}

  // Placeholder value.
  ListValue(om::ListType type, Location loc)
      : EvaluatorValue(type.getContext(), Kind::List, loc), type(type) {}

  const auto &getElements() const { return elements; }
  bool hasElements() const { return elementsInitialized; }

  /// Return the type of the value, which is a ListType.
  om::ListType getListType() const { return type; }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::List;
  }

private:
  om::ListType type;
  SmallVector<EvaluatorValuePtr> elements;
  bool elementsInitialized = false;
};

/// A composite Object, which has a type and fields.
class ObjectValue : public EvaluatorValue {
public:
  ObjectValue(om::ClassLike cls, ObjectFields fields, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls),
        fields(std::move(fields)), fieldsInitialized(true) {}

  // Placeholder value.
  ObjectValue(om::ClassLike cls, Location loc)
      : EvaluatorValue(cls.getContext(), Kind::Object, loc), cls(cls) {}

  om::ClassLike getClassOp() const { return cls; }
  const auto &getFields() const { return fields; }
  bool hasFields() const { return fieldsInitialized; }

  void setFields(llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> newFields) {
    fields = std::move(newFields);
    fieldsInitialized = true;
  }

  /// Return the type of the value, which is a ClassType.
  om::ClassType getObjectType() const {
    auto clsNonConst = const_cast<om::ClassLike &>(cls);
    return ClassType::get(clsNonConst.getContext(),
                          FlatSymbolRefAttr::get(clsNonConst.getSymNameAttr()));
  }

  Type getType() const { return getObjectType(); }

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Object;
  }

  /// Get a field of the Object by name.
  FailureOr<EvaluatorValuePtr> getField(StringAttr field);
  FailureOr<EvaluatorValuePtr> getField(StringRef field) {
    return getField(StringAttr::get(getContext(), field));
  }

  /// Get all the field names of the Object.
  ArrayAttr getFieldNames();

private:
  om::ClassLike cls;
  llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> fields;
  bool fieldsInitialized = false;
};

/// A Basepath value.
class BasePathValue : public EvaluatorValue {
public:
  BasePathValue(MLIRContext *context);

  /// Create a path value representing a basepath.
  BasePathValue(om::PathAttr path, Location loc);

  om::PathAttr getPath() const;

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::BasePath;
  }

private:
  om::PathAttr path;
};

/// A Path value.
class PathValue : public EvaluatorValue {
public:
  /// Create a path value representing a regular path.
  PathValue(om::TargetKindAttr targetKind, om::PathAttr path, StringAttr module,
            StringAttr ref, StringAttr field, Location loc);

  static PathValue getEmptyPath(Location loc);

  om::TargetKindAttr getTargetKind() const { return targetKind; }

  om::PathAttr getPath() const { return path; }

  StringAttr getModule() const { return module; }

  StringAttr getRef() const { return ref; }

  StringAttr getField() const { return field; }

  StringAttr getAsString() const;

  /// Implement LLVM RTTI.
  static bool classof(const EvaluatorValue *e) {
    return e->getKind() == Kind::Path;
  }

private:
  om::TargetKindAttr targetKind;
  om::PathAttr path;
  StringAttr module;
  StringAttr ref;
  StringAttr field;
};

} // namespace evaluator

using Object = evaluator::ObjectValue;
using EvaluatorValuePtr = evaluator::EvaluatorValuePtr;

SmallVector<EvaluatorValuePtr>
getEvaluatorValuesFromAttributes(MLIRContext *context,
                                 ArrayRef<Attribute> attributes);

/// An Evaluator, which is constructed with an IR module and can instantiate
/// Objects. Further refinement is expected.
class Evaluator {
public:
  /// Construct an Evaluator with an IR module.
  Evaluator(ModuleOp mod);

  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<evaluator::EvaluatorValuePtr>
  instantiate(StringAttr className, ArrayRef<EvaluatorValuePtr> actualParams);

  /// Get the Module this Evaluator is built from.
  mlir::ModuleOp getModule();

  FailureOr<evaluator::EvaluatorValuePtr> getPlaceholderValue(Type type,
                                                              Location loc);

  using ActualParameters = ArrayRef<evaluator::EvaluatorValuePtr>;

private:
  FailureOr<evaluator::EvaluatorValuePtr>
  instantiateImpl(StringAttr className,
                  ArrayRef<EvaluatorValuePtr> actualParams);

  FailureOr<EvaluatorValuePtr> getOrCreateValue(Value value, Location loc);

  /// Evaluate a Value in a Class body according to the small expression grammar
  /// described in the rationale document.
  FailureOr<EvaluatorValuePtr> evaluateValue(Value value, Location loc);

  /// Evaluator dispatch functions for the small expression grammar.
  FailureOr<EvaluatorValuePtr> evaluateParameter(BlockArgument formalParam,
                                                 Location loc);

  FailureOr<EvaluatorValuePtr> evaluateConstant(ConstantOp op, Location loc);

  /// Evaluate a class body with actual parameters.
  FailureOr<EvaluatorValuePtr> evaluateClass(StringAttr className,
                                             ActualParameters actualParams,
                                             Location loc);
  FailureOr<EvaluatorValuePtr> evaluateElaboratedObject(ElaboratedObjectOp op,
                                                        Location loc);
  FailureOr<EvaluatorValuePtr> evaluateListCreate(ListCreateOp op,
                                                  Location loc);
  FailureOr<evaluator::EvaluatorValuePtr>
  evaluateUnknownValue(UnknownValueOp op, Location loc);

  FailureOr<evaluator::EvaluatorValuePtr> createUnknownValue(Type type,
                                                             Location loc);

  /// The symbol table for the IR module the Evaluator was constructed with.
  /// Used to look up class definitions.
  SymbolTable symbolTable;

  /// Evaluator value storage for the current instantiation.
  DenseMap<Value, EvaluatorValuePtr> evaluatedValues;

#ifndef NDEBUG
  /// Current nesting depth for debug output indentation.
  unsigned debugNesting = 0;

  /// RAII helper to increment/decrement debugNesting.
  struct DebugNesting {
    unsigned &depth;
    DebugNesting(unsigned &depth) : depth(depth) { ++depth; }
    ~DebugNesting() { --depth; }
  };

  raw_ostream &dbgs(unsigned extra = 0) {
    return llvm::dbgs().indent(debugNesting * 2 + extra * 2);
  }

  llvm::indent indent(unsigned extra = 0) {
    return llvm::indent(debugNesting, 2) + extra;
  }
#endif
};

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &
operator<<(mlir::Diagnostic &diag,
           const evaluator::EvaluatorValue &evaluatorValue) {
  if (auto *attr = llvm::dyn_cast<evaluator::AttributeValue>(&evaluatorValue))
    diag << attr->getAttr();
  else if (auto *object =
               llvm::dyn_cast<evaluator::ObjectValue>(&evaluatorValue))
    diag << "Object(" << object->getType() << ")";
  else if (auto *list = llvm::dyn_cast<evaluator::ListValue>(&evaluatorValue))
    diag << "List(" << list->getType() << ")";
  else if (llvm::isa<evaluator::BasePathValue>(&evaluatorValue))
    diag << "BasePath()";
  else if (llvm::isa<evaluator::PathValue>(&evaluatorValue))
    diag << "Path()";
  else if (auto *unknown =
               llvm::dyn_cast<evaluator::UnknownValue>(&evaluatorValue))
    diag << "Unknown(" << unknown->getType() << ")";
  else
    assert(false && "unhandled evaluator value");
  return diag;
}

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &
operator<<(mlir::Diagnostic &diag, const EvaluatorValuePtr &evaluatorValue) {
  return diag << *evaluatorValue.get();
}

#ifndef NDEBUG
/// Helper to enable printing objects to raw_ostream (e.g., llvm::dbgs()).
/// Delegates to the Diagnostic overload via an intermediate string.
static inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const evaluator::EvaluatorValue &evaluatorValue) {
  std::string buf;
  llvm::raw_string_ostream ss(buf);
  mlir::Diagnostic diag(UnknownLoc::get(evaluatorValue.getContext()),
                        mlir::DiagnosticSeverity::Note);
  diag << evaluatorValue;
  ss << diag;
  return os << ss.str();
}

static inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &os, const EvaluatorValuePtr &evaluatorValue) {
  if (evaluatorValue)
    return os << *evaluatorValue.get();
  return os << "<null>";
}
#endif // NDEBUG

} // namespace om
} // namespace circt

#endif // CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
