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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

namespace {
using ResolvedValue = evaluator::ResolvedValue;
} // namespace

namespace circt::om::detail {
namespace {

/// Walk through reference values until we reach a non-reference value.
/// Return Pending if the chain ends at null. Return Failure if the chain loops.
ResolvedValue resolveReferenceValue(evaluator::EvaluatorValuePtr currentValue) {
  llvm::SmallPtrSet<evaluator::ReferenceValue *, 4> visited;
  if (!currentValue)
    return ResolvedValue::pending();

  while (auto *ref =
             llvm::dyn_cast<evaluator::ReferenceValue>(currentValue.get())) {
    if (!visited.insert(ref).second)
      return ResolvedValue::failure(currentValue);
    currentValue = ref->getValue();
    if (!currentValue)
      return ResolvedValue::pending();
  }

  return ResolvedValue::ready(std::move(currentValue));
}

} // namespace

} // namespace circt::om::detail

using circt::om::detail::resolveReferenceValue;

namespace {

struct InstanceContext {
  Operation *owner = nullptr;
  SmallVector<Value> actuals;
  DenseMap<Value, Value> importedValues;
  llvm::SmallDenseSet<Value, 8> activeImports;
};

struct ElaboratedObjectKey {
  Operation *objectOp = nullptr;
  const InstanceContext *parentContext = nullptr;
};

struct ElaboratedObjectKeyInfo {
  static inline ElaboratedObjectKey getEmptyKey() {
    return {DenseMapInfo<Operation *>::getEmptyKey(),
            DenseMapInfo<const InstanceContext *>::getEmptyKey()};
  }

  static inline ElaboratedObjectKey getTombstoneKey() {
    return {DenseMapInfo<Operation *>::getTombstoneKey(),
            DenseMapInfo<const InstanceContext *>::getTombstoneKey()};
  }

  static unsigned getHashValue(const ElaboratedObjectKey &key) {
    return llvm::hash_combine(key.objectOp, key.parentContext);
  }

  static bool isEqual(const ElaboratedObjectKey &lhs,
                      const ElaboratedObjectKey &rhs) {
    return lhs.objectOp == rhs.objectOp &&
           lhs.parentContext == rhs.parentContext;
  }
};

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
                         wrapperClass.getBodyBlock()->getArguments());
    opContexts[rootObject.getOperation()] = &wrapperContext;
    wrapperClass.updateFields({unknownLoc}, {rootObject.getResult()},
                              {builder.getStringAttr("root")});

    if (failed(rewriteWrapper()))
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
  template <typename OpTy>
  struct AttrFoldOpConversionPattern : OpConversionPattern<OpTy> {
    AttrFoldOpConversionPattern(MLIRContext *context)
        : OpConversionPattern<OpTy>(context) {}

    LogicalResult
    matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      bool hasUnknownOperand = false;
      SmallVector<Attribute> operandAttrs;
      operandAttrs.reserve(adaptor.getOperands().size());
      for (Value operand : adaptor.getOperands()) {
        if (operand.template getDefiningOp<UnknownValueOp>()) {
          hasUnknownOperand = true;
          break;
        }

        auto constant = operand.template getDefiningOp<ConstantOp>();
        if (!constant)
          return failure();
        operandAttrs.push_back(constant.getValue());
      }

      if (hasUnknownOperand) {
        rewriter.replaceOpWithNewOp<UnknownValueOp>(op, op.getType());
        return success();
      }

      SmallVector<OpFoldResult> foldResults;
      if (failed(op->fold(operandAttrs, foldResults)) ||
          foldResults.size() != 1)
        return failure();

      auto foldedAttr = dyn_cast<Attribute>(foldResults.front());
      if (!foldedAttr)
        return failure();

      auto typedAttr = dyn_cast<TypedAttr>(foldedAttr);
      if (!typedAttr)
        return failure();

      rewriter.replaceOpWithNewOp<ConstantOp>(op, typedAttr);
      return success();
    }
  };

  struct ObjectOpConversionPattern : OpConversionPattern<ObjectOp> {
    ObjectOpConversionPattern(MLIRContext *context, ScratchIRBuilder &scratch)
        : OpConversionPattern<ObjectOp>(context), scratch(scratch) {}

    LogicalResult
    matchAndRewrite(ObjectOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto converted =
          scratch.convertObject(op, adaptor.getActualParams(), rewriter);
      if (failed(converted))
        return failure();
      rewriter.replaceOp(op, converted.value());
      return success();
    }

    ScratchIRBuilder &scratch;
  };

  struct ObjectFieldOpConversionPattern : OpConversionPattern<ObjectFieldOp> {
    ObjectFieldOpConversionPattern(MLIRContext *context,
                                   ScratchIRBuilder &scratch)
        : OpConversionPattern<ObjectFieldOp>(context), scratch(scratch) {}

    LogicalResult
    matchAndRewrite(ObjectFieldOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto simplified = scratch.simplifyObjectField(
          adaptor.getObject(), op.getFieldAttr(), op.getType(), rewriter);
      if (failed(simplified))
        return failure();
      if (!*simplified)
        return failure();
      rewriter.replaceOp(op, **simplified);
      return success();
    }

    ScratchIRBuilder &scratch;
  };

  FailureOr<ClassLike> lookupSourceClassLike(StringAttr className) {
    if (auto classLike = sourceSymbols.lookup<ClassLike>(className))
      return classLike;
    return sourceModule.emitError("unknown class name ") << className;
  }

  FailureOr<ClassLike> lookupScratchClassLike(StringAttr className) {
    if (auto classLike = scratchSymbols.lookup<ClassLike>(className))
      return classLike;
    return scratchModule.emitError("unknown class name ") << className;
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

    SmallVector<std::string> paramNameStorage;
    SmallVector<StringRef> paramNames;
    SmallVector<Type> paramTypes;
    paramNameStorage.reserve(actualParams.size());
    paramNames.reserve(actualParams.size());
    paramTypes.reserve(actualParams.size());
    for (auto [index, actual] : llvm::enumerate(actualParams)) {
      paramNameStorage.push_back("arg" + std::to_string(index));
      paramNames.push_back(paramNameStorage.back());
      paramTypes.push_back(actual->getType());
    }

    wrapperClass = ClassOp::create(
        builder, loc, builder.getStringAttr(wrapperName),
        builder.getStrArrayAttr(paramNames), builder.getArrayAttr({}),
        builder.getDictionaryAttr({}));

    Block *body = &wrapperClass.getRegion().emplaceBlock();
    for (auto type : paramTypes)
      body->addArgument(type, loc);

    builder.setInsertionPointToEnd(body);
    ClassFieldsOp::create(builder, loc, ValueRange(), builder.getArrayAttr({}));

    wrapperContext.owner = wrapperClass.getOperation();
    llvm::append_range(wrapperContext.actuals, body->getArguments());
    for (auto [arg, actual] : llvm::zip(body->getArguments(), actualParams))
      actualInputs[arg] = actual;
    return success();
  }

  LogicalResult rewriteWrapper() {
    ConversionTarget target(*sourceModule.getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<ObjectOp>();
    target.addLegalOp<ElaboratedObjectOp>();
    auto needsAttrFold = [](Operation *op) {
      bool hasUnknownOperand = false;
      bool allConstantOperands = true;
      for (Value operand : op->getOperands()) {
        if (operand.getDefiningOp<UnknownValueOp>()) {
          hasUnknownOperand = true;
          break;
        }
        allConstantOperands &= operand.getDefiningOp<ConstantOp>() != nullptr;
      }
      return hasUnknownOperand || allConstantOperands;
    };
    target.addDynamicallyLegalOp<ObjectFieldOp>([](ObjectFieldOp op) {
      auto base = op.getObject();
      return !base.getDefiningOp<AnyCastOp>() &&
             !base.getDefiningOp<ElaboratedObjectOp>() &&
             !base.getDefiningOp<UnknownValueOp>();
    });
    target.addDynamicallyLegalOp<IntegerAddOp, IntegerMulOp, IntegerShrOp,
                                 IntegerShlOp, StringConcatOp, PropEqOp>(
        [&](Operation *op) { return !needsAttrFold(op); });

    RewritePatternSet patterns(sourceModule.getContext());
    patterns.add<ObjectOpConversionPattern, ObjectFieldOpConversionPattern>(
        sourceModule.getContext(), *this);
    patterns.add<AttrFoldOpConversionPattern<IntegerAddOp>,
                 AttrFoldOpConversionPattern<IntegerMulOp>,
                 AttrFoldOpConversionPattern<IntegerShrOp>,
                 AttrFoldOpConversionPattern<IntegerShlOp>,
                 AttrFoldOpConversionPattern<StringConcatOp>,
                 AttrFoldOpConversionPattern<PropEqOp>>(
        sourceModule.getContext());
    return applyFullConversion(wrapperClass, target, std::move(patterns));
  }

  Value createUnknownScratchValue(Type type, Location loc, OpBuilder &builder) {
    return UnknownValueOp::create(builder, loc, type).getResult();
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

  FailureOr<Operation *> cloneOperation(Operation *op, ArrayRef<Value> operands,
                                        InstanceContext &context,
                                        OpBuilder &builder) {
    builder.setInsertionPoint(wrapperClass.getFieldsOp());
    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addAttributes(op->getAttrs());
    state.addTypes(op->getResultTypes());
    state.addOperands(operands);
    Operation *cloned = builder.create(state);
    sourceOps[cloned] = op;
    opContexts[cloned] = &context;
    return success(cloned);
  }

  FailureOr<Value> materializeValue(Value value, InstanceContext &context,
                                    OpBuilder &builder) {
    if (auto arg = dyn_cast<BlockArgument>(value))
      return success(context.actuals[arg.getArgNumber()]);
    if (auto it = context.importedValues.find(value);
        it != context.importedValues.end())
      return success(it->second);
    if (!context.activeImports.insert(value).second)
      return mlir::emitError(value.getLoc(),
                             "rewritten OM evaluation contains a non-object "
                             "dataflow cycle");
    auto clearActive =
        llvm::scope_exit([&] { context.activeImports.erase(value); });

    Operation *op = value.getDefiningOp();
    SmallVector<Value> importedOperands;
    importedOperands.reserve(op->getNumOperands());
    for (auto operand : op->getOperands()) {
      auto importedOperand = materializeValue(operand, context, builder);
      if (failed(importedOperand))
        return failure();
      importedOperands.push_back(importedOperand.value());
    }

    auto clonedOp = cloneOperation(op, importedOperands, context, builder);
    if (failed(clonedOp))
      return failure();
    for (auto [sourceResult, clonedResult] :
         llvm::zip(op->getResults(), clonedOp.value()->getResults()))
      context.importedValues[sourceResult] = clonedResult;
    return success(context.importedValues.lookup(value));
  }

  FailureOr<Value> convertObject(ObjectOp objectOp, ValueRange actuals,
                                 ConversionPatternRewriter &rewriter) {
    InstanceContext *parentContext = &wrapperContext;
    if (auto it = opContexts.find(objectOp.getOperation());
        it != opContexts.end())
      parentContext = it->second;

    Operation *sourceObject = objectOp.getOperation();
    if (auto it = sourceOps.find(objectOp.getOperation());
        it != sourceOps.end())
      sourceObject = it->second;

    ElaboratedObjectKey key{sourceObject, parentContext};
    if (auto it = elaboratedObjects.find(key); it != elaboratedObjects.end())
      return success(it->second);

    auto classLike = lookupScratchClassLike(objectOp.getClassNameAttr());
    if (failed(classLike))
      return failure();

    if (isa<ClassExternOp>(*classLike)) {
      rewriter.setInsertionPoint(wrapperClass.getFieldsOp());
      auto unknown = createUnknownScratchValue(objectOp.getType(),
                                               objectOp.getLoc(), rewriter);
      elaboratedObjects[key] = unknown;
      return success(unknown);
    }

    auto fieldNames = classLike->getFieldNames();
    auto classOp = dyn_cast<ClassOp>(classLike.value().getOperation());
    assert(classOp && "non-external class should be a ClassOp");
    SmallVector<Value> fieldPlaceholders;
    fieldPlaceholders.reserve(fieldNames.size());
    for (auto [index, fieldName] : llvm::enumerate(fieldNames))
      fieldPlaceholders.push_back(createUnknownScratchValue(
          classOp.getFieldsOp().getFields()[index].getType(), objectOp.getLoc(),
          rewriter));

    rewriter.setInsertionPoint(wrapperClass.getFieldsOp());
    auto placeholder = ElaboratedObjectOp::create(
        rewriter, objectOp.getLoc(), *classLike, fieldPlaceholders);
    elaboratedObjects[key] = placeholder.getResult();

    auto instanceContext = std::make_unique<InstanceContext>();
    instanceContext->owner = sourceObject;
    llvm::append_range(instanceContext->actuals, actuals);
    InstanceContext *instanceContextPtr = instanceContext.get();
    ownedContexts.push_back(std::move(instanceContext));

    SmallVector<Value> fieldValues;
    fieldValues.reserve(fieldNames.size());
    for (auto fieldValue : classOp.getFieldsOp().getOperands()) {
      auto importedField =
          materializeValue(fieldValue, *instanceContextPtr, rewriter);
      if (failed(importedField))
        return fieldValue.getDefiningOp()->emitError(
                   "failed to import elaborated field")
               << " for " << objectOp.getClassNameAttr();
      fieldValues.push_back(importedField.value());
    }
    placeholder->setOperands(fieldValues);

    for (auto propertyAssert : classOp.getOps<PropertyAssertOp>()) {
      auto importedCondition = materializeValue(propertyAssert.getCondition(),
                                                *instanceContextPtr, rewriter);
      if (failed(importedCondition))
        return propertyAssert.emitError(
            "failed to import property assertion condition");
      rewriter.setInsertionPoint(wrapperClass.getFieldsOp());
      PropertyAssertOp::create(rewriter, propertyAssert.getLoc(),
                               importedCondition.value(),
                               propertyAssert.getMessage());
    }

    return success(placeholder.getResult());
  }

  FailureOr<std::optional<Value>> simplifyObjectField(Value base,
                                                      FlatSymbolRefAttr field,
                                                      Type resultType,
                                                      OpBuilder &builder) {
    if (auto anyCast = base.getDefiningOp<AnyCastOp>())
      return std::optional<Value>(ObjectFieldOp::create(
          builder, anyCast.getLoc(), resultType, anyCast.getInput(), field));

    if (base.getDefiningOp<UnknownValueOp>())
      return std::optional<Value>(
          createUnknownScratchValue(resultType, base.getLoc(), builder));

    auto elaboratedObject = base.getDefiningOp<ElaboratedObjectOp>();
    if (!elaboratedObject)
      return std::optional<Value>();

    auto classLike =
        lookupScratchClassLike(elaboratedObject.getClassNameAttr());
    if (failed(classLike))
      return failure();

    auto fieldNames = classLike->getFieldNames();
    auto fieldIt = llvm::find(fieldNames, field.getAttr());
    if (fieldIt == fieldNames.end())
      return elaboratedObject.emitError("field ") << field << " does not exist";
    size_t index = std::distance(fieldNames.begin(), fieldIt);
    auto classOp = dyn_cast<ClassOp>(classLike.value().getOperation());
    assert(classOp && "elaborated object must reference a ClassOp");
    assert(classOp.getFieldsOp().getFields()[index].getType() == resultType &&
           "field access result type should match class field type");
    return std::optional<Value>(elaboratedObject.getFieldValues()[index]);
  }

  FailureOr<EvaluatorValuePtr> exportValue(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value))
      return actualInputs.lookup(arg);
    if (auto it = exportedValues.find(value); it != exportedValues.end())
      return it->second;
    if (!activeExports.insert(value).second)
      return emitError(value.getLoc(), "rewritten OM evaluation contains a "
                                       "non-object dataflow cycle");
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
                                   "evaluator scratch IR");
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
  InstanceContext wrapperContext;
  DenseMap<Value, EvaluatorValuePtr> actualInputs;
  DenseMap<Operation *, Operation *> sourceOps;
  DenseMap<Operation *, InstanceContext *> opContexts;
  DenseMap<ElaboratedObjectKey, Value, ElaboratedObjectKeyInfo>
      elaboratedObjects;
  SmallVector<std::unique_ptr<InstanceContext>> ownedContexts;
  DenseMap<Value, EvaluatorValuePtr> exportedValues;
  llvm::SmallDenseSet<Value, 16> activeExports;
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
      .Case<AttributeValue, ObjectValue, ListValue, ReferenceValue,
            BasePathValue, PathValue>([](auto v) { return v->finalizeImpl(); });
}

Type circt::om::evaluator::EvaluatorValue::getType() const {
  return llvm::TypeSwitch<const EvaluatorValue *, Type>(this)
      .Case<AttributeValue>([](auto *attr) -> Type { return attr->getType(); })
      .Case<ObjectValue>([](auto *object) { return object->getObjectType(); })
      .Case<ListValue>([](auto *list) { return list->getListType(); })
      .Case<ReferenceValue>([](auto *ref) { return ref->getValueType(); })
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
// ReferenceValue
//===----------------------------------------------------------------------===//

FailureOr<EvaluatorValuePtr>
circt::om::evaluator::ReferenceValue::getStrippedValue() const {
  auto resolved = resolveReferenceValue(value);
  switch (resolved.state) {
  case ResolutionState::Ready:
    return success(resolved.value);
  case ResolutionState::Pending:
    return mlir::emitError(getLoc(), "reference value is not resolved");
  case ResolutionState::Failure:
    return mlir::emitError(getLoc(), "reference value contains a cycle");
  }
  llvm_unreachable("unknown resolution state");
}

LogicalResult circt::om::evaluator::ReferenceValue::finalizeImpl() {
  auto resolved = resolveReferenceValue(value);
  if (resolved.state != ResolutionState::Ready)
    return failure();
  value = std::move(resolved.value);
  // the stripped value also needs to be finalized
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
