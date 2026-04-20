//===- Evaluator.cpp - Object Model dialect evaluator ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Object Model dialect Evaluator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/Transforms/ElaborationTransform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

namespace circt::om::detail {
namespace {} // namespace

} // namespace circt::om::detail

namespace {

class ScratchIRBuilder {
public:
  explicit ScratchIRBuilder(Evaluator &evaluator)
      : sourceModule(evaluator.getModule()), sourceSymbols(sourceModule),
        scratchModule(cast<ModuleOp>(sourceModule->clone())),
        scratchSymbols(scratchModule), builder(sourceModule.getContext()) {}

  FailureOr<EvaluatorValuePtr> run(StringAttr rootClassName,
                                   ArrayRef<EvaluatorValuePtr> actualParams) {
    auto *ctx = sourceModule.getContext();
    auto unknownLoc = UnknownLoc::get(ctx);
    auto rootClass = scratchSymbols.lookup<ClassLike>(rootClassName);
    if (!rootClass)
      return sourceModule.emitError("unknown class name ") << rootClassName;
    if (failed(verifyActualParameters(rootClass, actualParams)))
      return failure();

    if (failed(createWrapperClass(actualParams, unknownLoc)))
      return failure();

    builder.setInsertionPoint(wrapperClass.getFieldsOp());
    auto rootType = ClassType::get(ctx, FlatSymbolRefAttr::get(rootClassName));
    auto rootObject =
        ObjectOp::create(builder, unknownLoc, rootType, rootClassName,
                         importedActualValues);
    wrapperClass.updateFields({unknownLoc}, {rootObject.getResult()},
                              {builder.getStringAttr("root")});

    if (failed(applyElaborationTransform(wrapperClass, scratchSymbols)))
      return failure();

    if (failed(checkPropertyAssertions()))
      return failure();

    auto exportedRoot =
        exportValue(wrapperClass.getFieldsOp().getFields().front());
    if (failed(exportedRoot))
      return failure();
    if (failed(exportedRoot.value()->finalize()))
      return wrapperClass.emitError(
          "failed to finalize rewritten OM evaluation");
    return exportedRoot;
  }

private:
  FailureOr<ClassLike> lookupSourceClassLike(StringAttr className) {
    if (auto classLike = sourceSymbols.lookup<ClassLike>(className))
      return classLike;
    return sourceModule.emitError("unknown class name ") << className;
  }

  LogicalResult
  verifyActualParameters(ClassLike classLike,
                         ArrayRef<EvaluatorValuePtr> actualParams) {
    auto formalParamNames =
        classLike.getFormalParamNames().getAsRange<StringAttr>();
    auto formalParamTypes = classLike.getBodyBlock()->getArgumentTypes();

    if (actualParams.size() != formalParamTypes.size()) {
      auto error = classLike.emitError("actual parameter list length (")
                   << actualParams.size() << ") does not match formal "
                   << "parameter list length (" << formalParamTypes.size()
                   << ")";
      auto &diag = error.attachNote() << "actual parameters: ";
      bool isFirst = true;
      for (const auto &param : actualParams) {
        if (isFirst)
          isFirst = false;
        else
          diag << ", ";
        diag << param;
      }
      error.attachNote(classLike.getLoc())
          << "formal parameters: " << formalParamTypes;
      return failure();
    }

    for (auto [actualParam, formalParamName, formalParamType] :
         llvm::zip(actualParams, formalParamNames, formalParamTypes)) {
      if (!actualParam || !actualParam.get())
        return classLike.emitError("actual parameter for ")
               << formalParamName << " is null";
      if (isa<AnyType>(formalParamType))
        continue;
      if (actualParam->getType() != formalParamType) {
        auto error = classLike.emitError("actual parameter for ")
                     << formalParamName << " has invalid type";
        error.attachNote() << "actual parameter: " << *actualParam;
        error.attachNote() << "format parameter type: " << formalParamType;
        return failure();
      }
    }
    return success();
  }

  LogicalResult createWrapperClass(ArrayRef<EvaluatorValuePtr> actualParams,
                                   Location loc) {
    builder.setInsertionPointToEnd(scratchModule.getBody());

    std::string wrapperName = "__om_evaluator_wrapper";
    unsigned suffix = 0;
    while (scratchSymbols.lookup<ClassLike>(builder.getStringAttr(wrapperName)))
      wrapperName = "__om_evaluator_wrapper_" + std::to_string(++suffix);

    wrapperClass = ClassOp::create(
        builder, loc, builder.getStringAttr(wrapperName),
        builder.getStrArrayAttr({}), builder.getArrayAttr({}),
        builder.getDictionaryAttr({}));

    Block *body = &wrapperClass.getRegion().emplaceBlock();
    importedActualValues.clear();
    importedActualValues.reserve(actualParams.size());
    builder.setInsertionPointToEnd(body);
    for (auto actual : actualParams) {
      if (actual->isUnknown()) {
        importedActualValues.push_back(
            UnknownValueOp::create(builder, loc, actual->getType()).getResult());
        continue;
      }
      if (auto *attrValue =
              dyn_cast<evaluator::AttributeValue>(actual.get())) {
        importedActualValues.push_back(
            ConstantOp::create(builder, loc,
                               cast<TypedAttr>(attrValue->getAttr()))
                .getResult());
        continue;
      }
      auto arg = body->addArgument(actual->getType(), loc);
      importedActualValues.push_back(arg);
      actualInputs[arg] = actual;
    }
    builder.setInsertionPointToEnd(body);
    ClassFieldsOp::create(builder, loc, ValueRange(), builder.getArrayAttr({}));
    return success();
  }

  FailureOr<EvaluatorValuePtr> createUnknownRuntimeValue(Type type,
                                                         Location loc) {
    using namespace circt::om::evaluator;
    auto result =
        TypeSwitch<Type, FailureOr<EvaluatorValuePtr>>(type)
            .Case([&](ListType listType) -> FailureOr<EvaluatorValuePtr> {
              return success(
                  std::make_shared<evaluator::ListValue>(listType, loc));
            })
            .Case([&](ClassType classType) -> FailureOr<EvaluatorValuePtr> {
              auto classLike =
                  lookupSourceClassLike(classType.getClassName().getAttr());
              if (failed(classLike))
                return failure();
              return success(
                  std::make_shared<evaluator::ObjectValue>(*classLike, loc));
            })
            .Case([&](FrozenBasePathType type) -> FailureOr<EvaluatorValuePtr> {
              return success(std::make_shared<evaluator::BasePathValue>(
                  type.getContext()));
            })
            .Case([&](FrozenPathType type) -> FailureOr<EvaluatorValuePtr> {
              return success(std::make_shared<evaluator::PathValue>(
                  evaluator::PathValue::getEmptyPath(loc)));
            })
            .Default([&](Type type) -> FailureOr<EvaluatorValuePtr> {
              return success(evaluator::AttributeValue::get(type));
            });
    if (failed(result))
      return failure();
    result.value()->markUnknown();
    return result;
  }

  FailureOr<EvaluatorValuePtr> exportValue(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value))
      return actualInputs.lookup(arg);
    if (auto it = exportedValues.find(value); it != exportedValues.end())
      return it->second;
    if (!activeExports.insert(value).second)
      return emitError(value.getLoc(),
                       "failed to finalize evaluation. Probably the class "
                       "contains a dataflow cycle");
    auto eraseActive = llvm::scope_exit([&] { activeExports.erase(value); });

    Operation *op = value.getDefiningOp();
    auto result =
        TypeSwitch<Operation *, FailureOr<EvaluatorValuePtr>>(op)
            .Case([&](ConstantOp op) -> FailureOr<EvaluatorValuePtr> {
              return evaluator::AttributeValue::get(op.getValue());
            })
            .Case([&](UnknownValueOp op) -> FailureOr<EvaluatorValuePtr> {
              return createUnknownRuntimeValue(op.getType(), op.getLoc());
            })
            .Case([&](AnyCastOp op) -> FailureOr<EvaluatorValuePtr> {
              return exportValue(op.getInput());
            })
            .Case([&](ElaboratedObjectOp op) -> FailureOr<EvaluatorValuePtr> {
              auto sourceClass = lookupSourceClassLike(op.getClassNameAttr());
              if (failed(sourceClass))
                return failure();
              auto objectValue = std::make_shared<evaluator::ObjectValue>(
                  *sourceClass, op.getLoc());
              exportedValues[value] = objectValue;

              llvm::SmallDenseMap<StringAttr, EvaluatorValuePtr> fields;
              auto fieldNames = sourceClass->getFieldNames();
              for (auto [fieldName, fieldValue] :
                   llvm::zip(fieldNames, op.getFieldValues())) {
                auto exportedField = exportValue(fieldValue);
                if (failed(exportedField))
                  return failure();
                fields[cast<StringAttr>(fieldName)] = exportedField.value();
              }
              objectValue->setFields(std::move(fields));
              return std::static_pointer_cast<evaluator::EvaluatorValue>(
                  objectValue);
            })
            .Case([&](ObjectFieldOp op) -> FailureOr<EvaluatorValuePtr> {
              if (auto elaboratedObject =
                      op.getObject().getDefiningOp<ElaboratedObjectOp>()) {
                auto sourceClass =
                    lookupSourceClassLike(elaboratedObject.getClassNameAttr());
                if (failed(sourceClass))
                  return failure();
                auto fieldNames = sourceClass->getFieldNames();
                auto fieldIt =
                    llvm::find(fieldNames, op.getFieldAttr().getAttr());
                if (fieldIt == fieldNames.end())
                  return op.emitError("field ") << op.getFieldAttr()
                                                << " does not exist";
                return exportValue(elaboratedObject.getFieldValues()[std::distance(
                    fieldNames.begin(), fieldIt)]);
              }
              auto baseValue = exportValue(op.getObject());
              if (failed(baseValue))
                return failure();
              if (baseValue.value()->isUnknown())
                return createUnknownRuntimeValue(op.getType(), op.getLoc());
              auto *object =
                  dyn_cast<evaluator::ObjectValue>(baseValue.value().get());
              if (!object)
                return emitError(
                    op.getLoc(),
                    "expected object while resolving object.field");
              auto fieldValue = object->getField(op.getFieldAttr().getAttr());
              if (failed(fieldValue))
                return failure();
              return fieldValue.value();
            })
            .Case([&](ListCreateOp op) -> FailureOr<EvaluatorValuePtr> {
              SmallVector<EvaluatorValuePtr> elements;
              elements.reserve(op.getInputs().size());
              for (auto input : op.getInputs()) {
                auto element = exportValue(input);
                if (failed(element))
                  return failure();
                if (element.value()->isUnknown())
                  return createUnknownRuntimeValue(op.getType(), op.getLoc());
                elements.push_back(element.value());
              }
              return std::static_pointer_cast<evaluator::EvaluatorValue>(
                  std::make_shared<evaluator::ListValue>(
                      op.getType(), std::move(elements), op.getLoc()));
            })
            .Case([&](ListConcatOp op) -> FailureOr<EvaluatorValuePtr> {
              SmallVector<EvaluatorValuePtr> elements;
              for (auto input : op.getSubLists()) {
                auto listValue = exportValue(input);
                if (failed(listValue))
                  return failure();
                if (listValue.value()->isUnknown())
                  return createUnknownRuntimeValue(op.getType(), op.getLoc());
                auto *list =
                    dyn_cast<evaluator::ListValue>(listValue.value().get());
                if (!list)
                  return op.emitError("expected list operand");
                llvm::append_range(elements, list->getElements());
              }
              return std::static_pointer_cast<evaluator::EvaluatorValue>(
                  std::make_shared<evaluator::ListValue>(
                      op.getType(), std::move(elements), op.getLoc()));
            })
            .Case(
                [&](FrozenBasePathCreateOp op) -> FailureOr<EvaluatorValuePtr> {
                  auto baseValue = exportValue(op.getBasePath());
                  if (failed(baseValue))
                    return failure();
                  if (baseValue.value()->isUnknown())
                    return createUnknownRuntimeValue(op.getType(), op.getLoc());
                  auto *basePath = dyn_cast<evaluator::BasePathValue>(
                      baseValue.value().get());
                  if (!basePath)
                    return op.emitError("expected base path operand");
                  auto result = std::make_shared<evaluator::BasePathValue>(
                      op.getPathAttr(), op.getLoc());
                  result->setBasepath(*basePath);
                  return std::static_pointer_cast<evaluator::EvaluatorValue>(
                      result);
                })
            .Case([&](FrozenPathCreateOp op) -> FailureOr<EvaluatorValuePtr> {
              auto baseValue = exportValue(op.getBasePath());
              if (failed(baseValue))
                return failure();
              if (baseValue.value()->isUnknown())
                return createUnknownRuntimeValue(op.getType(), op.getLoc());
              auto *basePath =
                  dyn_cast<evaluator::BasePathValue>(baseValue.value().get());
              if (!basePath)
                return op.emitError("expected base path operand");
              auto result = std::make_shared<evaluator::PathValue>(
                  op.getTargetKindAttr(), op.getPathAttr(), op.getModuleAttr(),
                  op.getRefAttr(), op.getFieldAttr(), op.getLoc());
              result->setBasepath(*basePath);
              return std::static_pointer_cast<evaluator::EvaluatorValue>(
                  result);
            })
            .Case([&](FrozenEmptyPathOp op) -> FailureOr<EvaluatorValuePtr> {
              return std::static_pointer_cast<evaluator::EvaluatorValue>(
                  std::make_shared<evaluator::PathValue>(
                      evaluator::PathValue::getEmptyPath(op.getLoc())));
            })
            .Default([&](Operation *op) -> FailureOr<EvaluatorValuePtr> {
              return op->emitError("unsupported operation in rewritten OM "
                                   "evaluator scratch IR: ")
                     << op->getName();
            });

    if (failed(result))
      return failure();
    exportedValues[value] = result.value();
    return result;
  }

  LogicalResult checkPropertyAssertions() {
    for (auto propertyAssert : wrapperClass.getOps<PropertyAssertOp>()) {
      auto conditionValue = exportValue(propertyAssert.getCondition());
      if (failed(conditionValue))
        return failure();
      if (conditionValue.value()->isUnknown())
        continue;

      auto *attrValue =
          dyn_cast<evaluator::AttributeValue>(conditionValue.value().get());
      if (!attrValue)
        return emitError(propertyAssert.getLoc(),
                         "expected property assertion condition to evaluate "
                         "to an attribute");

      bool isFalse = false;
      if (auto boolAttr = dyn_cast<BoolAttr>(attrValue->getAttr()))
        isFalse = !boolAttr.getValue();
      else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(attrValue->getAttr()))
        isFalse = intAttr.getValue().isZero();
      else
        return emitError(propertyAssert.getLoc(),
                         "expected BoolAttr or IntegerAttr");

      if (isFalse)
        return emitError(propertyAssert.getLoc(),
                         "OM property assertion failed: ")
               << propertyAssert.getMessage();
    }
    return success();
  }

  ModuleOp sourceModule;
  SymbolTable sourceSymbols;
  ModuleOp scratchModule;
  SymbolTable scratchSymbols;
  OpBuilder builder;
  ClassOp wrapperClass;
  DenseMap<Value, EvaluatorValuePtr> actualInputs;
  DenseMap<Value, EvaluatorValuePtr> exportedValues;
  llvm::SmallDenseSet<Value, 16> activeExports;
  SmallVector<Value> importedActualValues;
};

} // namespace

/// Construct an Evaluator with an IR module.
circt::om::Evaluator::Evaluator(ModuleOp mod) : symbolTable(mod) {}

/// Get the Module this Evaluator is built from.
ModuleOp circt::om::Evaluator::getModule() {
  return cast<ModuleOp>(symbolTable.getOp());
}

SmallVector<evaluator::EvaluatorValuePtr>
circt::om::getEvaluatorValuesFromAttributes(MLIRContext *context,
                                            ArrayRef<Attribute> attributes) {
  SmallVector<evaluator::EvaluatorValuePtr> values;
  values.reserve(attributes.size());
  for (auto attr : attributes)
    values.push_back(evaluator::AttributeValue::get(cast<TypedAttr>(attr)));
  return values;
}

LogicalResult circt::om::evaluator::EvaluatorValue::finalize() {
  using namespace evaluator;
  // Early return if already finalized.
  if (finalized)
    return success();
  // Enable the flag to avoid infinite recursions.
  finalized = true;
  assert(isSettled());
  return llvm::TypeSwitch<EvaluatorValue *, LogicalResult>(this)
      .Case<AttributeValue, ObjectValue, ListValue, BasePathValue, PathValue>(
          [](auto v) { return v->finalizeImpl(); });
}

Type circt::om::evaluator::EvaluatorValue::getType() const {
  return llvm::TypeSwitch<const EvaluatorValue *, Type>(this)
      .Case<AttributeValue>([](auto *attr) -> Type { return attr->getType(); })
      .Case<ObjectValue>([](auto *object) { return object->getObjectType(); })
      .Case<ListValue>([](auto *list) { return list->getListType(); })
      .Case<BasePathValue>(
          [this](auto *tuple) { return FrozenBasePathType::get(ctx); })
      .Case<PathValue>(
          [this](auto *tuple) { return FrozenPathType::get(ctx); });
}

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<evaluator::EvaluatorValue>>
circt::om::Evaluator::instantiate(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  ScratchIRBuilder scratchBuilder(*this);
  return scratchBuilder.run(className, actualParams);
}

//===----------------------------------------------------------------------===//
// ObjectValue
//===----------------------------------------------------------------------===//

/// Get a field of the Object by name.
FailureOr<EvaluatorValuePtr>
circt::om::evaluator::ObjectValue::getField(StringAttr name) {
  auto field = fields.find(name);
  if (field == fields.end())
    return cls.emitError("field ") << name << " does not exist";
  return success(fields[name]);
}

/// Get an ArrayAttr with the names of the fields in the Object. Sort the fields
/// so there is always a stable order.
ArrayAttr circt::om::Object::getFieldNames() {
  SmallVector<Attribute> fieldNames;
  for (auto &f : fields)
    fieldNames.push_back(f.first);

  llvm::sort(fieldNames, [](Attribute a, Attribute b) {
    return cast<StringAttr>(a).getValue() < cast<StringAttr>(b).getValue();
  });

  return ArrayAttr::get(cls.getContext(), fieldNames);
}

LogicalResult circt::om::evaluator::ObjectValue::finalizeImpl() {
  for (auto &&[e, value] : fields)
    if (failed(finalizeEvaluatorValue(value)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ListValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::ListValue::finalizeImpl() {
  for (auto &value : elements) {
    if (failed(finalizeEvaluatorValue(value)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// BasePathValue
//===----------------------------------------------------------------------===//

evaluator::BasePathValue::BasePathValue(MLIRContext *context)
    : EvaluatorValue(context, Kind::BasePath, UnknownLoc::get(context)),
      path(PathAttr::get(context, {})) {
  markSettled();
}

evaluator::BasePathValue::BasePathValue(PathAttr path, Location loc)
    : EvaluatorValue(path.getContext(), Kind::BasePath, loc), path(path) {}

PathAttr evaluator::BasePathValue::getPath() const {
  assert(isSettled());
  return path;
}

void evaluator::BasePathValue::setBasepath(const BasePathValue &basepath) {
  assert(!isSettled());
  auto newPath = llvm::to_vector(basepath.path.getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markSettled();
}

//===----------------------------------------------------------------------===//
// PathValue
//===----------------------------------------------------------------------===//

evaluator::PathValue::PathValue(TargetKindAttr targetKind, PathAttr path,
                                StringAttr module, StringAttr ref,
                                StringAttr field, Location loc)
    : EvaluatorValue(loc.getContext(), Kind::Path, loc), targetKind(targetKind),
      path(path), module(module), ref(ref), field(field) {}

evaluator::PathValue evaluator::PathValue::getEmptyPath(Location loc) {
  PathValue path(nullptr, nullptr, nullptr, nullptr, nullptr, loc);
  path.markSettled();
  return path;
}

StringAttr evaluator::PathValue::getAsString() const {
  // If the module is null, then this is a path to a deleted object.
  if (!targetKind)
    return StringAttr::get(getContext(), "OMDeleted:");
  SmallString<64> result;
  switch (targetKind.getValue()) {
  case TargetKind::DontTouch:
    result += "OMDontTouchedReferenceTarget";
    break;
  case TargetKind::Instance:
    result += "OMInstanceTarget";
    break;
  case TargetKind::MemberInstance:
    result += "OMMemberInstanceTarget";
    break;
  case TargetKind::MemberReference:
    result += "OMMemberReferenceTarget";
    break;
  case TargetKind::Reference:
    result += "OMReferenceTarget";
    break;
  }
  result += ":~";
  if (!path.getPath().empty())
    result += path.getPath().front().module;
  else
    result += module.getValue();
  result += '|';
  for (const auto &elt : path) {
    result += elt.module.getValue();
    result += '/';
    result += elt.instance.getValue();
    result += ':';
  }
  if (!module.getValue().empty())
    result += module.getValue();
  if (!ref.getValue().empty()) {
    result += '>';
    result += ref.getValue();
  }
  if (!field.getValue().empty())
    result += field.getValue();
  return StringAttr::get(field.getContext(), result);
}

void evaluator::PathValue::setBasepath(const BasePathValue &basepath) {
  assert(!isSettled());
  auto newPath = llvm::to_vector(basepath.getPath().getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markSettled();
}

//===----------------------------------------------------------------------===//
// AttributeValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::AttributeValue::setAttr(Attribute attr) {
  if (cast<TypedAttr>(attr).getType() != this->type)
    return mlir::emitError(getLoc(), "cannot set AttributeValue of type ")
           << this->type << " to Attribute " << attr;
  if (isSettled())
    return mlir::emitError(getLoc(),
                           "cannot set AttributeValue that is already settled");
  this->attr = attr;
  markSettled();
  return success();
}

LogicalResult circt::om::evaluator::AttributeValue::finalizeImpl() {
  if (!isSettled())
    return mlir::emitError(
        getLoc(), "cannot finalize AttributeValue that is not settled");
  return success();
}

std::shared_ptr<evaluator::EvaluatorValue>
circt::om::evaluator::AttributeValue::get(Attribute attr, LocationAttr loc) {
  auto type = cast<TypedAttr>(attr).getType();
  auto *context = type.getContext();
  if (!loc)
    loc = UnknownLoc::get(context);

  // Special handling for ListType to create proper ListValue objects instead of
  // AttributeValue objects.
  if (auto listType = dyn_cast<circt::om::ListType>(type)) {
    SmallVector<EvaluatorValuePtr> elements;
    auto listAttr = cast<om::ListAttr>(attr);
    auto values = getEvaluatorValuesFromAttributes(
        listAttr.getContext(), listAttr.getElements().getValue());
    elements.append(values.begin(), values.end());
    auto list = std::make_shared<evaluator::ListValue>(listType, elements, loc);
    return list;
  }

  return std::shared_ptr<AttributeValue>(
      new AttributeValue(PrivateTag{}, attr, loc));
}

std::shared_ptr<evaluator::EvaluatorValue>
circt::om::evaluator::AttributeValue::get(Type type, LocationAttr loc) {
  auto *context = type.getContext();
  if (!loc)
    loc = UnknownLoc::get(context);

  // Special handling for ListType to create proper ListValue objects instead of
  // AttributeValue objects.
  if (auto listType = dyn_cast<circt::om::ListType>(type))
    return std::make_shared<evaluator::ListValue>(listType, loc);
  // Create the AttributeValue with the private tag
  return std::shared_ptr<AttributeValue>(
      new AttributeValue(PrivateTag{}, type, loc));
}
