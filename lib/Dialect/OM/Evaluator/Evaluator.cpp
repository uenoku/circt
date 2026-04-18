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
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

namespace {

enum class ResolutionState { Ready, Pending, Failure };

struct ResolvedValue {
  ResolutionState state;
  evaluator::EvaluatorValuePtr value;

  static ResolvedValue ready(evaluator::EvaluatorValuePtr value) {
    return {ResolutionState::Ready, std::move(value)};
  }
  static ResolvedValue pending() { return {ResolutionState::Pending, nullptr}; }
  static ResolvedValue failure() { return {ResolutionState::Failure, nullptr}; }
};

static ResolvedValue
resolveReferenceValue(evaluator::EvaluatorValuePtr currentValue) {
  llvm::SmallPtrSet<evaluator::ReferenceValue *, 4> visited;
  if (!currentValue)
    return ResolvedValue::pending();

  while (auto *ref =
             llvm::dyn_cast<evaluator::ReferenceValue>(currentValue.get())) {
    if (!visited.insert(ref).second)
      return ResolvedValue::failure();
    currentValue = ref->getValue();
    if (!currentValue)
      return ResolvedValue::pending();
  }

  return ResolvedValue::ready(std::move(currentValue));
}

static ResolvedValue inspectValue(evaluator::EvaluatorValuePtr currentValue) {
  if (!currentValue || !currentValue->isFullyEvaluated())
    return ResolvedValue::pending();

  auto resolved = resolveReferenceValue(currentValue);
  if (resolved.state != ResolutionState::Ready)
    return resolved;
  if (!resolved.value->isFullyEvaluated())
    return ResolvedValue::pending();

  return resolved;
}

static bool
isUnknownValue(evaluator::EvaluatorValuePtr value,
               evaluator::EvaluatorValuePtr resolvedValue = nullptr) {
  return (value && value->isUnknown()) ||
         (resolvedValue && resolvedValue->isUnknown());
}

static bool isUnknownValue(evaluator::EvaluatorValuePtr value,
                           const ResolvedValue &resolved) {
  return isUnknownValue(std::move(value), resolved.value);
}

template <typename T>
static T *getResolvedValueAs(const ResolvedValue &resolved,
                             const char *assertMessage) {
  auto *typedValue = dyn_cast<T>(resolved.value.get());
  assert(typedValue && assertMessage);
  return typedValue;
}

template <typename AttrT>
static AttrT getResolvedAttrAs(const ResolvedValue &resolved,
                               const char *assertMessage) {
  auto *attrValue =
      getResolvedValueAs<evaluator::AttributeValue>(resolved, assertMessage);
  AttrT attr = attrValue->getAs<AttrT>();
  assert(attr && assertMessage);
  return attr;
}

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
  assert(isFullyEvaluated());
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

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::getPartiallyEvaluatedValue(Type type, Location loc) {
  using namespace circt::om::evaluator;

  return TypeSwitch<mlir::Type, FailureOr<evaluator::EvaluatorValuePtr>>(type)
      .Case([&](circt::om::ListType type) {
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::ListValue>(type, loc);
        return success(result);
      })
      .Case([&](circt::om::ClassType type)
                -> FailureOr<evaluator::EvaluatorValuePtr> {
        auto classDef =
            symbolTable.lookup<ClassLike>(type.getClassName().getValue());
        if (!classDef)
          return symbolTable.getOp()->emitError("unknown class name ")
                 << type.getClassName();

        // Create an ObjectValue for both ClassOp and ClassExternOp
        evaluator::EvaluatorValuePtr result =
            std::make_shared<evaluator::ObjectValue>(classDef, loc);

        return success(result);
      })
      .Case([&](circt::om::StringType type) {
        evaluator::EvaluatorValuePtr result =
            evaluator::AttributeValue::get(type, loc);
        return success(result);
      })
      .Default([&](auto type) { return failure(); });
}

FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::getOrCreateValue(
    Value value, ActualParameters actualParams, Location loc) {
  auto it = objects.find({value, actualParams});
  if (it != objects.end()) {
    auto evalVal = it->second;
    evalVal->setLocIfUnknown(loc);
    return evalVal;
  }

  FailureOr<evaluator::EvaluatorValuePtr> result =
      TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
          .Case([&](BlockArgument arg) {
            auto val = (*actualParams)[arg.getArgNumber()];
            val->setLoc(loc);
            return val;
          })
          .Case([&](OpResult result) {
            return TypeSwitch<Operation *,
                              FailureOr<evaluator::EvaluatorValuePtr>>(
                       result.getDefiningOp())
                .Case([&](ConstantOp op) {
                  return evaluateConstant(op, actualParams, loc);
                })
                .Case([&](IntegerBinaryArithmeticOp op) {
                  // Create a partially evaluated AttributeValue of
                  // om::IntegerType in case we need to delay evaluation.
                  evaluator::EvaluatorValuePtr result =
                      evaluator::AttributeValue::get(op.getResult().getType(),
                                                     loc);
                  return success(result);
                })
                .Case<ObjectFieldOp>([&](auto op) {
                  // Create a reference value since the value pointed by object
                  // field op is not created yet.
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::ReferenceValue>(
                          value.getType(), loc);
                  return success(result);
                })
                .Case<AnyCastOp>([&](AnyCastOp op) {
                  return getOrCreateValue(op.getInput(), actualParams, loc);
                })
                .Case<FrozenBasePathCreateOp>([&](FrozenBasePathCreateOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::BasePathValue>(
                          op.getPathAttr(), loc);
                  return success(result);
                })
                .Case<FrozenPathCreateOp>([&](FrozenPathCreateOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::PathValue>(
                          op.getTargetKindAttr(), op.getPathAttr(),
                          op.getModuleAttr(), op.getRefAttr(),
                          op.getFieldAttr(), loc);
                  return success(result);
                })
                .Case<FrozenEmptyPathOp>([&](FrozenEmptyPathOp op) {
                  evaluator::EvaluatorValuePtr result =
                      std::make_shared<evaluator::PathValue>(
                          evaluator::PathValue::getEmptyPath(loc));
                  return success(result);
                })
                .Case([&](BinaryEqualityOp op) {
                  evaluator::EvaluatorValuePtr result =
                      evaluator::AttributeValue::get(op.getResult().getType(),
                                                     loc);
                  return success(result);
                })
                .Case<ListCreateOp, ListConcatOp, StringConcatOp,
                      ObjectFieldOp>([&](auto op) {
                  return getPartiallyEvaluatedValue(op.getType(), loc);
                })
                .Case<ObjectOp>([&](auto op) {
                  return getPartiallyEvaluatedValue(op.getType(), op.getLoc());
                })
                .Case<UnknownValueOp>(
                    [&](auto op) { return evaluateUnknownValue(op, loc); })
                .Default([&](Operation *op) {
                  auto error = op->emitError("unable to evaluate value");
                  error.attachNote() << "value: " << value;
                  return error;
                });
          });
  if (failed(result))
    return result;

  objects[{value, actualParams}] = result.value();
  return result;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(StringAttr className,
                                             ActualParameters actualParams,
                                             Location loc,
                                             ObjectKey instanceKey) {
  auto classDef = symbolTable.lookup<ClassLike>(className);
  if (!classDef)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  // If this is an external class, create an ObjectValue and mark it unknown
  if (isa<ClassExternOp>(classDef)) {
    evaluator::EvaluatorValuePtr result =
        std::make_shared<evaluator::ObjectValue>(classDef, loc);
    result->markUnknown();
    return result;
  }

  // Otherwise, it's a regular class, proceed normally
  ClassOp cls = cast<ClassOp>(classDef);

  auto formalParamNames = cls.getFormalParamNames().getAsRange<StringAttr>();
  auto formalParamTypes = cls.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameters are the right size and types for this class.
  if (actualParams->size() != formalParamTypes.size()) {
    auto error = cls.emitError("actual parameter list length (")
                 << actualParams->size() << ") does not match formal "
                 << "parameter list length (" << formalParamTypes.size() << ")";
    auto &diag = error.attachNote() << "actual parameters: ";
    // FIXME: `diag << actualParams` doesn't work for some reason.
    bool isFirst = true;
    for (const auto &param : *actualParams) {
      if (isFirst)
        isFirst = false;
      else
        diag << ", ";
      diag << param;
    }
    error.attachNote(cls.getLoc()) << "formal parameters: " << formalParamTypes;
    return error;
  }

  // Verify the actual parameter types match.
  for (auto [actualParam, formalParamName, formalParamType] :
       llvm::zip(*actualParams, formalParamNames, formalParamTypes)) {
    if (!actualParam || !actualParam.get())
      return cls.emitError("actual parameter for ")
             << formalParamName << " is null";

    // Subtyping: if formal param is any type, any actual param may be passed.
    if (isa<AnyType>(formalParamType))
      continue;

    Type actualParamType = actualParam->getType();

    assert(actualParamType && "actualParamType must be non-null!");

    if (actualParamType != formalParamType) {
      auto error = cls.emitError("actual parameter for ")
                   << formalParamName << " has invalid type";
      error.attachNote() << "actual parameter: " << *actualParam;
      error.attachNote() << "format parameter type: " << formalParamType;
      return error;
    }
  }

  // Instantiate the fields.
  evaluator::ObjectFields fields;

  auto *context = cls.getContext();
  for (auto &op : cls.getOps())
    for (auto result : op.getResults()) {
      // Allocate the value, with unknown loc. It will be later set when
      // evaluating the fields.
      if (failed(
              getOrCreateValue(result, actualParams, UnknownLoc::get(context))))
        return failure();
      // Add to the worklist.
      worklist.push({result, actualParams});
    }

  auto fieldNames = cls.getFieldNames();
  auto operands = cls.getFieldsOp()->getOperands();
  for (size_t i = 0; i < fieldNames.size(); ++i) {
    auto name = fieldNames[i];
    auto value = operands[i];
    auto fieldLoc = cls.getFieldLocByIndex(i);
    FailureOr<evaluator::EvaluatorValuePtr> result =
        evaluateValue(value, actualParams, fieldLoc);
    if (failed(result))
      return result;

    fields[cast<StringAttr>(name)] = result.value();
  }

  // Evaluate property assertions.
  for (auto assertOp : cls.getOps<PropertyAssertOp>())
    if (failed(evaluatePropertyAssert(assertOp, actualParams)))
      return failure();

  // If the there is an instance, we must update the object value.
  if (instanceKey.first) {
    auto result =
        getOrCreateValue(instanceKey.first, instanceKey.second, loc).value();
    auto *object = llvm::cast<evaluator::ObjectValue>(result.get());
    object->setFields(std::move(fields));
    return result;
  }

  // If it's external call, just allocate new ObjectValue.
  evaluator::EvaluatorValuePtr result =
      std::make_shared<evaluator::ObjectValue>(cls, fields, loc);
  return result;
}

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<evaluator::EvaluatorValue>>
circt::om::Evaluator::instantiate(
    StringAttr className, ArrayRef<evaluator::EvaluatorValuePtr> actualParams) {
  auto classDef = symbolTable.lookup<ClassLike>(className);
  if (!classDef)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  // If this is an external class, create an ObjectValue and mark it unknown
  if (isa<ClassExternOp>(classDef)) {
    evaluator::EvaluatorValuePtr result =
        std::make_shared<evaluator::ObjectValue>(
            classDef, UnknownLoc::get(classDef.getContext()));
    result->markUnknown();
    return result;
  }

  // Otherwise, it's a regular class, proceed normally
  ClassOp cls = cast<ClassOp>(classDef);

  auto parameters =
      std::make_unique<SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>(
          actualParams);

  actualParametersBuffers.push_back(std::move(parameters));

  auto loc = cls.getLoc();
  auto result = evaluateObjectInstance(
      className, actualParametersBuffers.back().get(), loc);

  if (failed(result))
    return failure();

  // `evaluateObjectInstance` has populated the worklist. Continue evaluations
  // unless there is a partially evaluated value.
  while (!worklist.empty()) {
    auto [value, args] = worklist.front();
    worklist.pop();

    auto result = evaluateValue(value, args, loc);

    if (failed(result))
      return failure();

    // It's possible that the value is not fully evaluated.
    if (!result.value()->isFullyEvaluated())
      worklist.push({value, args});
  }

  auto &object = result.value();
  // Finalize the value. This will eliminate intermidiate ReferenceValue used as
  // a placeholder in the initialization.
  if (failed(object->finalize()))
    return cls.emitError() << "failed to finalize evaluation. Probably the "
                              "class contains a dataflow cycle";
  return object;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateValue(Value value, ActualParameters actualParams,
                                    Location loc) {
  auto evaluatorValue = getOrCreateValue(value, actualParams, loc).value();

  // Return if the value is already evaluated.
  if (evaluatorValue->isFullyEvaluated())
    return evaluatorValue;

  return llvm::TypeSwitch<Value, FailureOr<evaluator::EvaluatorValuePtr>>(value)
      .Case([&](BlockArgument arg) {
        return evaluateParameter(arg, actualParams, loc);
      })
      .Case([&](OpResult result) {
        return TypeSwitch<Operation *, FailureOr<evaluator::EvaluatorValuePtr>>(
                   result.getDefiningOp())
            .Case([&](ConstantOp op) {
              return evaluateConstant(op, actualParams, loc);
            })
            .Case([&](IntegerBinaryArithmeticOp op) {
              return evaluateIntegerBinaryArithmetic(op, actualParams, loc);
            })
            .Case([&](ObjectOp op) {
              return evaluateObjectInstance(op, actualParams);
            })
            .Case([&](ObjectFieldOp op) {
              return evaluateObjectField(op, actualParams, loc);
            })
            .Case([&](ListCreateOp op) {
              return evaluateListCreate(op, actualParams, loc);
            })
            .Case([&](ListConcatOp op) {
              return evaluateListConcat(op, actualParams, loc);
            })
            .Case([&](StringConcatOp op) {
              return evaluateStringConcat(op, actualParams, loc);
            })
            .Case([&](BinaryEqualityOp op) {
              return evaluateBinaryEquality(op, actualParams, loc);
            })
            .Case([&](AnyCastOp op) {
              return evaluateValue(op.getInput(), actualParams, loc);
            })
            .Case([&](FrozenBasePathCreateOp op) {
              return evaluateBasePathCreate(op, actualParams, loc);
            })
            .Case([&](FrozenPathCreateOp op) {
              return evaluatePathCreate(op, actualParams, loc);
            })
            .Case([&](FrozenEmptyPathOp op) {
              return evaluateEmptyPath(op, actualParams, loc);
            })
            .Case<UnknownValueOp>([&](UnknownValueOp op) {
              return evaluateUnknownValue(op, loc);
            })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return error;
            });
      });
}

/// Evaluator dispatch function for parameters.
FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateParameter(
    BlockArgument formalParam, ActualParameters actualParams, Location loc) {
  auto val = (*actualParams)[formalParam.getArgNumber()];
  val->setLoc(loc);
  return success(val);
}

/// Evaluator dispatch function for constants.
FailureOr<circt::om::evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateConstant(ConstantOp op,
                                       ActualParameters actualParams,
                                       Location loc) {
  // For list constants, create ListValue.
  return success(om::evaluator::AttributeValue::get(op.getValue(), loc));
}

// Evaluator dispatch function for integer binary arithmetic.
FailureOr<EvaluatorValuePtr>
circt::om::Evaluator::evaluateIntegerBinaryArithmetic(
    IntegerBinaryArithmeticOp op, ActualParameters actualParams, Location loc) {
  // Get the op's EvaluatorValue handle, in case it hasn't been evaluated yet.
  auto handle = getOrCreateValue(op.getResult(), actualParams, loc);

  // If it's fully evaluated, we can return it.
  if (handle.value()->isFullyEvaluated())
    return handle;

  // Evaluate operands if necessary, and return the partially evaluated value if
  // they aren't ready.
  auto lhsResult = evaluateValue(op.getLhs(), actualParams, loc);
  if (failed(lhsResult))
    return lhsResult;
  auto rhsResult = evaluateValue(op.getRhs(), actualParams, loc);
  if (failed(rhsResult))
    return rhsResult;

  auto lhsValue = inspectValue(lhsResult.value());
  switch (lhsValue.state) {
  case ResolutionState::Pending:
    return handle;
  case ResolutionState::Failure:
    return op->emitError("failed to resolve integer arithmetic lhs");
  case ResolutionState::Ready:
    break;
  }

  auto rhsValue = inspectValue(rhsResult.value());
  switch (rhsValue.state) {
  case ResolutionState::Pending:
    return handle;
  case ResolutionState::Failure:
    return op->emitError("failed to resolve integer arithmetic rhs");
  case ResolutionState::Ready:
    break;
  }

  // Check if any operand is unknown and propagate the unknown flag.
  if (isUnknownValue(lhsResult.value(), lhsValue) ||
      isUnknownValue(rhsResult.value(), rhsValue)) {
    handle.value()->markUnknown();
    return handle;
  }

  om::IntegerAttr lhs = getResolvedAttrAs<om::IntegerAttr>(
      lhsValue, "expected om::IntegerAttr for IntegerBinaryArithmeticOp lhs");
  om::IntegerAttr rhs = getResolvedAttrAs<om::IntegerAttr>(
      rhsValue, "expected om::IntegerAttr for IntegerBinaryArithmeticOp rhs");

  // Extend values if necessary to match bitwidth. Most interesting arithmetic
  // on APSInt asserts that both operands are the same bitwidth, but the
  // IntegerAttrs we are working with may have used the smallest necessary
  // bitwidth to represent the number they hold, and won't necessarily match.
  APSInt lhsVal = lhs.getValue().getAPSInt();
  APSInt rhsVal = rhs.getValue().getAPSInt();
  if (lhsVal.getBitWidth() > rhsVal.getBitWidth())
    rhsVal = rhsVal.extend(lhsVal.getBitWidth());
  else if (rhsVal.getBitWidth() > lhsVal.getBitWidth())
    lhsVal = lhsVal.extend(rhsVal.getBitWidth());

  // Perform arbitrary precision signed integer binary arithmetic.
  FailureOr<APSInt> result = op.evaluateIntegerOperation(lhsVal, rhsVal);

  if (failed(result))
    return op->emitError("failed to evaluate integer operation");

  // Package the result as a new om::IntegerAttr.
  MLIRContext *ctx = op->getContext();
  auto resultAttr =
      om::IntegerAttr::get(ctx, mlir::IntegerAttr::get(ctx, result.value()));

  // Finalize the op result value.
  auto *handleValue = cast<evaluator::AttributeValue>(handle.value().get());
  auto resultStatus = handleValue->setAttr(resultAttr);
  if (failed(resultStatus))
    return resultStatus;

  auto finalizeStatus = handleValue->finalize();
  if (failed(finalizeStatus))
    return finalizeStatus;

  return handle;
}

/// Evaluator dispatch function for property assertions.
LogicalResult
circt::om::Evaluator::evaluatePropertyAssert(PropertyAssertOp op,
                                             ActualParameters actualParams) {
  auto loc = op.getLoc();

  // Evaluate the condition, returning early if it isn't ready yet.
  auto condResult = evaluateValue(op.getCondition(), actualParams, loc);
  if (failed(condResult))
    return failure();

  auto resolvedCond = inspectValue(condResult.value());
  switch (resolvedCond.state) {
  case ResolutionState::Pending:
    return success();
  case ResolutionState::Failure:
    return op.emitError("failed to resolve property assertion condition");
  case ResolutionState::Ready:
    break;
  }

  // If the condition is unknown, skip silently (best-effort).
  if (isUnknownValue(condResult.value(), resolvedCond))
    return success();

  auto condAttr =
      getResolvedValueAs<evaluator::AttributeValue>(
          resolvedCond, "expected attribute value for property assertion")
          ->getAttr();

  bool isFalse = false;
  if (auto boolAttr = dyn_cast<BoolAttr>(condAttr))
    isFalse = !boolAttr.getValue();
  else if (auto intAttr = dyn_cast<mlir::IntegerAttr>(condAttr))
    isFalse = intAttr.getValue().isZero();
  else
    return op.emitError("expected BoolAttr or mlir::IntegerAttr");

  if (isFalse)
    return op.emitError("OM property assertion failed: ") << op.getMessage();

  return success();
}

/// Evaluator dispatch function for Object instances.
FailureOr<circt::om::Evaluator::ActualParameters>
circt::om::Evaluator::createParametersFromOperands(
    ValueRange range, ActualParameters actualParams, Location loc) {
  // Create an unique storage to store parameters.
  auto parameters = std::make_unique<
      SmallVector<std::shared_ptr<evaluator::EvaluatorValue>>>();

  // Collect operands' evaluator values in the current instantiation context.
  for (auto input : range) {
    auto inputResult = getOrCreateValue(input, actualParams, loc);
    if (failed(inputResult))
      return failure();

    auto inputValue = inputResult.value();
    if (isa<evaluator::ReferenceValue>(inputValue.get())) {
      auto evaluatedInput = evaluateValue(input, actualParams, loc);
      if (failed(evaluatedInput))
        return failure();
      inputValue = evaluatedInput.value();
    }
    parameters->push_back(inputValue);
  }

  actualParametersBuffers.push_back(std::move(parameters));
  return actualParametersBuffers.back().get();
}

/// Evaluator dispatch function for Object instances.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectInstance(ObjectOp op,
                                             ActualParameters actualParams) {
  auto loc = op.getLoc();
  auto key = ObjectKey{op, actualParams};
  if (isFullyEvaluated(key))
    return getOrCreateValue(op, actualParams, loc);
  if (!activeObjectInstances.insert(key).second)
    return getOrCreateValue(op, actualParams, loc);
  auto clearActiveObject =
      llvm::scope_exit([&] { activeObjectInstances.erase(key); });

  auto params =
      createParametersFromOperands(op.getOperands(), actualParams, loc);
  if (failed(params))
    return failure();
  return evaluateObjectInstance(op.getClassNameAttr(), params.value(), loc,
                                {op, actualParams});
}

/// Evaluator dispatch function for Object fields.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateObjectField(ObjectFieldOp op,
                                          ActualParameters actualParams,
                                          Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  FailureOr<evaluator::EvaluatorValuePtr> currentObjectResult =
      evaluateValue(op.getObject(), actualParams, loc);
  if (failed(currentObjectResult))
    return currentObjectResult;

  auto result = currentObjectResult.value();

  auto objectFieldValue = getOrCreateValue(op, actualParams, loc).value();

  auto setUnknownFieldValue = [&]() -> FailureOr<evaluator::EvaluatorValuePtr> {
    auto unknownField = createUnknownValue(op.getResult().getType(), loc);
    if (failed(unknownField))
      return unknownField;

    if (auto *ref =
            llvm::dyn_cast<evaluator::ReferenceValue>(objectFieldValue.get()))
      ref->setValue(unknownField.value());

    objectFieldValue->markUnknown();
    return objectFieldValue;
  };

  auto resolvedObject = inspectValue(result);
  switch (resolvedObject.state) {
  case ResolutionState::Pending:
    return objectFieldValue;
  case ResolutionState::Failure:
    return op.emitError("failed to resolve object field base");
  case ResolutionState::Ready:
    break;
  }

  if (isUnknownValue(result, resolvedObject))
    return setUnknownFieldValue();

  auto *currentObject =
      llvm::dyn_cast<evaluator::ObjectValue>(resolvedObject.value.get());
  if (!currentObject)
    return op.emitError("resolved object field base is not an object");

  // Iteratively access nested fields through the path until we reach the final
  // field in the path.
  evaluator::EvaluatorValuePtr finalField;
  auto fieldPath = op.getFieldPath().getAsRange<FlatSymbolRefAttr>();
  for (auto it = fieldPath.begin(), end = fieldPath.end(); it != end; ++it) {
    auto field = *it;
    // `currentObject` might no be fully evaluated.
    if (!currentObject->getFields().contains(field.getAttr()))
      return objectFieldValue;

    auto currentField = currentObject->getField(field.getAttr());
    finalField = currentField.value();
    if (std::next(it) == end)
      continue;

    auto nextObject = inspectValue(finalField);
    switch (nextObject.state) {
    case ResolutionState::Pending:
      return objectFieldValue;
    case ResolutionState::Failure:
      return op.emitError("failed to resolve nested object field path");
    case ResolutionState::Ready:
      break;
    }
    if (isUnknownValue(finalField, nextObject))
      return setUnknownFieldValue();

    currentObject =
        llvm::dyn_cast<evaluator::ObjectValue>(nextObject.value.get());
    if (!currentObject)
      return op.emitError("resolved nested object field path is not an object");
  }

  // Update the reference.
  llvm::cast<evaluator::ReferenceValue>(objectFieldValue.get())
      ->setValue(finalField);

  // Return the field being accessed.
  return objectFieldValue;
}

/// Evaluator dispatch function for List creation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateListCreate(ListCreateOp op,
                                         ActualParameters actualParams,
                                         Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, actualParams, loc);
  bool hasUnknown = false;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams, loc);
    if (failed(result))
      return result;

    auto evaluatedOperand = inspectValue(result.value());
    switch (evaluatedOperand.state) {
    case ResolutionState::Pending:
      return list;
    case ResolutionState::Failure:
      return op.emitError("failed to resolve list_create operand");
    case ResolutionState::Ready:
      break;
    }
    // Check if any operand is unknown.
    if (isUnknownValue(result.value(), evaluatedOperand))
      hasUnknown = true;
    values.push_back(result.value());
  }

  // Set the list elements (this also marks the list as fully evaluated).
  llvm::cast<evaluator::ListValue>(list.value().get())
      ->setElements(std::move(values));

  // If any operand is unknown, mark the list as unknown.
  // markUnknown() checks if already fully evaluated before calling
  // markFullyEvaluated().
  if (hasUnknown)
    list.value()->markUnknown();

  return list;
}

/// Evaluator dispatch function for List concatenation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateListConcat(ListConcatOp op,
                                         ActualParameters actualParams,
                                         Location loc) {
  // Evaluate the List concat op itself, in case it hasn't been evaluated yet.
  SmallVector<evaluator::EvaluatorValuePtr> values;
  auto list = getOrCreateValue(op, actualParams, loc);

  bool hasUnknown = false;
  for (auto operand : op.getOperands()) {
    auto result = evaluateValue(operand, actualParams, loc);
    if (failed(result))
      return result;

    auto subList = inspectValue(result.value());
    switch (subList.state) {
    case ResolutionState::Pending:
      return list;
    case ResolutionState::Failure:
      return op->emitError("failed to resolve list_concat operand");
    case ResolutionState::Ready:
      break;
    }

    // Check if any operand is unknown.
    if (isUnknownValue(result.value(), subList))
      hasUnknown = true;

    auto *subListValue = getResolvedValueAs<evaluator::ListValue>(
        subList, "expected list value for list_concat operand");

    // Append each EvaluatorValue from the sublist.
    for (const auto &subValue : subListValue->getElements())
      values.push_back(subValue);
  }

  // Return the concatenated list.
  llvm::cast<evaluator::ListValue>(list.value().get())
      ->setElements(std::move(values));

  // If any operand is unknown, mark the result as unknown.
  // markUnknown() checks if already fully evaluated before calling
  // markFullyEvaluated().
  if (hasUnknown)
    list.value()->markUnknown();

  return list;
}

/// Evaluator dispatch function for String concatenation.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateStringConcat(StringConcatOp op,
                                           ActualParameters actualParams,
                                           Location loc) {
  // Get the op's EvaluatorValue handle, in case it hasn't been evaluated yet.
  auto handle = getOrCreateValue(op.getResult(), actualParams, loc);
  if (failed(handle))
    return handle;

  // If it's fully evaluated, we can return it.
  if (handle.value()->isFullyEvaluated())
    return handle;

  // Evaluate all operands and concatenate them.
  std::string result;
  for (auto operand : op.getOperands()) {
    auto operandResult = evaluateValue(operand, actualParams, loc);
    if (failed(operandResult))
      return operandResult;

    auto resolvedOperand = inspectValue(operandResult.value());
    switch (resolvedOperand.state) {
    case ResolutionState::Pending:
      return handle;
    case ResolutionState::Failure:
      return op->emitError("failed to resolve string_concat operand");
    case ResolutionState::Ready:
      break;
    }
    if (isUnknownValue(operandResult.value(), resolvedOperand)) {
      handle.value()->markUnknown();
      return handle;
    }

    StringAttr str = getResolvedAttrAs<StringAttr>(
        resolvedOperand, "expected StringAttr for StringConcatOp operand");
    result += str.getValue().str();
  }

  // Create the concatenated string attribute.
  auto resultStr = StringAttr::get(result, op.getResult().getType());

  // Finalize the op result value.
  auto *handleValue = cast<evaluator::AttributeValue>(handle.value().get());
  auto resultStatus = handleValue->setAttr(resultStr);
  if (failed(resultStatus))
    return resultStatus;

  auto finalizeStatus = handleValue->finalize();
  if (failed(finalizeStatus))
    return finalizeStatus;

  return handle;
}

// Evaluator dispatch function for binary property equality operations.
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateBinaryEquality(BinaryEqualityOp op,
                                             ActualParameters actualParams,
                                             Location loc) {
  // Get the op's EvaluatorValue handle, in case it hasn't been evaluated yet.
  auto handle = getOrCreateValue(op.getResult(), actualParams, loc);
  if (failed(handle))
    return handle;

  // If it's fully evaluated, we can return it.
  if (handle.value()->isFullyEvaluated())
    return handle;

  // Evaluate both operands, returning the partially evaluated handle if either
  // isn't ready yet.
  auto lhsResult = evaluateValue(op.getLhs(), actualParams, loc);
  if (failed(lhsResult))
    return lhsResult;
  auto rhsResult = evaluateValue(op.getRhs(), actualParams, loc);
  if (failed(rhsResult))
    return rhsResult;

  auto lhsValue = inspectValue(lhsResult.value());
  switch (lhsValue.state) {
  case ResolutionState::Pending:
    return handle;
  case ResolutionState::Failure:
    return op->emitError("failed to resolve prop.eq lhs");
  case ResolutionState::Ready:
    break;
  }

  auto rhsValue = inspectValue(rhsResult.value());
  switch (rhsValue.state) {
  case ResolutionState::Pending:
    return handle;
  case ResolutionState::Failure:
    return op->emitError("failed to resolve prop.eq rhs");
  case ResolutionState::Ready:
    break;
  }

  // Check if any operand is unknown and propagate the unknown flag.
  if (isUnknownValue(lhsResult.value(), lhsValue) ||
      isUnknownValue(rhsResult.value(), rhsValue)) {
    handle.value()->markUnknown();
    return handle;
  }

  mlir::Attribute lhs =
      getResolvedValueAs<evaluator::AttributeValue>(
          lhsValue, "expected attribute value for BinaryEqualityOp lhs")
          ->getAttr();
  mlir::Attribute rhs =
      getResolvedValueAs<evaluator::AttributeValue>(
          rhsValue, "expected attribute value for BinaryEqualityOp rhs")
          ->getAttr();
  assert(lhs && rhs && "expected attribute for BinaryEqualityOp operands");

  // Perform the binary equality operation.
  FailureOr<mlir::Attribute> result = op.evaluateBinaryEquality(lhs, rhs);
  if (failed(result))
    return op->emitError("failed to evaluate binary equality operation");

  // Finalize the op result value.
  auto *handleValue = cast<evaluator::AttributeValue>(handle.value().get());
  auto resultStatus = handleValue->setAttr(*result);
  if (failed(resultStatus))
    return resultStatus;

  auto finalizeStatus = handleValue->finalize();
  if (failed(finalizeStatus))
    return finalizeStatus;

  return handle;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateBasePathCreate(FrozenBasePathCreateOp op,
                                             ActualParameters actualParams,
                                             Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  auto *path = llvm::cast<evaluator::BasePathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), actualParams, loc);
  if (failed(result))
    return result;

  auto basePath = inspectValue(result.value());
  switch (basePath.state) {
  case ResolutionState::Pending:
    return valueResult;
  case ResolutionState::Failure:
    return op.emitError("failed to resolve frozenbasepath_create operand");
  case ResolutionState::Ready:
    break;
  }

  // If the base path is unknown, mark the result as unknown.
  if (isUnknownValue(result.value(), basePath)) {
    valueResult->markUnknown();
    return valueResult;
  }

  auto *basePathValue =
      dyn_cast<evaluator::BasePathValue>(basePath.value.get());
  if (!basePathValue)
    return op.emitError(
        "resolved frozenbasepath_create operand is not a base path");
  path->setBasepath(*basePathValue);
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluatePathCreate(FrozenPathCreateOp op,
                                         ActualParameters actualParams,
                                         Location loc) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  auto *path = llvm::cast<evaluator::PathValue>(valueResult.get());
  auto result = evaluateValue(op.getBasePath(), actualParams, loc);
  if (failed(result))
    return result;

  auto basePath = inspectValue(result.value());
  switch (basePath.state) {
  case ResolutionState::Pending:
    return valueResult;
  case ResolutionState::Failure:
    return op.emitError("failed to resolve frozenpath_create operand");
  case ResolutionState::Ready:
    break;
  }

  // If the base path is unknown, mark the result as unknown.
  if (isUnknownValue(result.value(), basePath)) {
    valueResult->markUnknown();
    return valueResult;
  }

  auto *basePathValue =
      dyn_cast<evaluator::BasePathValue>(basePath.value.get());
  if (!basePathValue)
    return op.emitError(
        "resolved frozenpath_create operand is not a base path");
  path->setBasepath(*basePathValue);
  return valueResult;
}

FailureOr<evaluator::EvaluatorValuePtr> circt::om::Evaluator::evaluateEmptyPath(
    FrozenEmptyPathOp op, ActualParameters actualParams, Location loc) {
  auto valueResult = getOrCreateValue(op, actualParams, loc).value();
  return valueResult;
}

/// Create an unknown value of the specified type
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::createUnknownValue(Type type, Location loc) {
  using namespace circt::om::evaluator;

  // Create an unknown value of the appropriate type by switching on the type
  auto result =
      TypeSwitch<Type, FailureOr<EvaluatorValuePtr>>(type)
          .Case([&](ListType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty list
            return success(std::make_shared<ListValue>(type, loc));
          })
          .Case([&](ClassType type) -> FailureOr<EvaluatorValuePtr> {
            // Look up the class definition
            auto classDef =
                symbolTable.lookup<ClassLike>(type.getClassName().getValue());
            if (!classDef)
              return symbolTable.getOp()->emitError("unknown class name ")
                     << type.getClassName();

            // Create an ObjectValue for both ClassOp and ClassExternOp
            return success(std::make_shared<ObjectValue>(classDef, loc));
          })
          .Case([&](FrozenBasePathType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty basepath
            return success(std::make_shared<BasePathValue>(type.getContext()));
          })
          .Case([&](FrozenPathType type) -> FailureOr<EvaluatorValuePtr> {
            // Create an empty path
            return success(
                std::make_shared<PathValue>(PathValue::getEmptyPath(loc)));
          })
          .Default([&](Type type) -> FailureOr<EvaluatorValuePtr> {
            // For all other types (primitives like integer, string,
            // etc.), create an AttributeValue
            return success(AttributeValue::get(type, LocationAttr(loc)));
          });

  // Mark the result as unknown if successful
  if (succeeded(result))
    result->get()->markUnknown();

  return result;
}

/// Evaluate an unknown value
FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::evaluateUnknownValue(UnknownValueOp op, Location loc) {
  return createUnknownValue(op.getType(), loc);
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
  markFullyEvaluated();
}

evaluator::BasePathValue::BasePathValue(PathAttr path, Location loc)
    : EvaluatorValue(path.getContext(), Kind::BasePath, loc), path(path) {}

PathAttr evaluator::BasePathValue::getPath() const {
  assert(isFullyEvaluated());
  return path;
}

void evaluator::BasePathValue::setBasepath(const BasePathValue &basepath) {
  assert(!isFullyEvaluated());
  auto newPath = llvm::to_vector(basepath.path.getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markFullyEvaluated();
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
  path.markFullyEvaluated();
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
  assert(!isFullyEvaluated());
  auto newPath = llvm::to_vector(basepath.getPath().getPath());
  auto oldPath = path.getPath();
  newPath.append(oldPath.begin(), oldPath.end());
  path = PathAttr::get(path.getContext(), newPath);
  markFullyEvaluated();
}

//===----------------------------------------------------------------------===//
// AttributeValue
//===----------------------------------------------------------------------===//

LogicalResult circt::om::evaluator::AttributeValue::setAttr(Attribute attr) {
  if (cast<TypedAttr>(attr).getType() != this->type)
    return mlir::emitError(getLoc(), "cannot set AttributeValue of type ")
           << this->type << " to Attribute " << attr;
  if (isFullyEvaluated())
    return mlir::emitError(
        getLoc(),
        "cannot set AttributeValue that has already been fully evaluated");
  this->attr = attr;
  markFullyEvaluated();
  return success();
}

LogicalResult circt::om::evaluator::AttributeValue::finalizeImpl() {
  if (!isFullyEvaluated())
    return mlir::emitError(
        getLoc(), "cannot finalize AttributeValue that is not fully evaluated");
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
