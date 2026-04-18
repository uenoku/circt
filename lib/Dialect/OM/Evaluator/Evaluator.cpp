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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <optional>

#define DEBUG_TYPE "om-evaluator"

using namespace mlir;
using namespace circt::om;

namespace {

using ResolutionState = evaluator::ResolutionState;
using ResolvedValue = evaluator::ResolvedValue;

static ResolvedValue
resolveReferenceValue(evaluator::EvaluatorValuePtr currentValue) {
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

static ResolvedValue inspectValue(evaluator::EvaluatorValuePtr currentValue) {
  if (!currentValue || !currentValue->isFullyEvaluated())
    return ResolvedValue::pending(std::move(currentValue));

  auto resolved = resolveReferenceValue(currentValue);
  if (resolved.state != ResolutionState::Ready)
    return {resolved.state, std::move(currentValue)};
  if (!resolved.value->isFullyEvaluated())
    return ResolvedValue::pending(std::move(currentValue));

  return ResolvedValue::ready(std::move(currentValue));
}

struct ReadyValue {
  evaluator::EvaluatorValuePtr handle;
  evaluator::EvaluatorValuePtr resolved;

  bool isUnknown() const {
    return (handle && handle->isUnknown()) ||
           (resolved && resolved->isUnknown());
  }

  evaluator::AttributeValue *getAttrValue(const char *assertMessage) const {
    assert(resolved && assertMessage);
    auto *typedValue = dyn_cast<evaluator::AttributeValue>(resolved.get());
    assert(typedValue && assertMessage);
    return typedValue;
  }

  evaluator::ObjectValue *getObjectValue(const char *assertMessage) const {
    assert(resolved && assertMessage);
    auto *typedValue = dyn_cast<evaluator::ObjectValue>(resolved.get());
    assert(typedValue && assertMessage);
    return typedValue;
  }

  evaluator::ListValue *getListValue(const char *assertMessage) const {
    assert(resolved && assertMessage);
    auto *typedValue = dyn_cast<evaluator::ListValue>(resolved.get());
    assert(typedValue && assertMessage);
    return typedValue;
  }

  evaluator::BasePathValue *getBasePathValue(const char *assertMessage) const {
    assert(resolved && assertMessage);
    auto *typedValue = dyn_cast<evaluator::BasePathValue>(resolved.get());
    assert(typedValue && assertMessage);
    return typedValue;
  }

  mlir::Attribute getAttr(const char *assertMessage) const {
    return getAttrValue(assertMessage)->getAttr();
  }

  StringAttr getStringAttr(const char *assertMessage) const {
    StringAttr attr = dyn_cast<StringAttr>(getAttr(assertMessage));
    assert(attr && assertMessage);
    return attr;
  }

  circt::om::IntegerAttr getIntegerAttr(const char *assertMessage) const {
    circt::om::IntegerAttr attr =
        dyn_cast<circt::om::IntegerAttr>(getAttr(assertMessage));
    assert(attr && assertMessage);
    return attr;
  }
};

static ReadyValue materializeReadyValue(const ResolvedValue &resolved,
                                        const char *assertMessage) {
  assert(resolved.state == ResolutionState::Ready && assertMessage);
  auto stripped = resolveReferenceValue(resolved.value);
  assert(stripped.state == ResolutionState::Ready && assertMessage);
  assert(stripped.value && stripped.value->isFullyEvaluated() && assertMessage);
  return {resolved.value, stripped.value};
}

static std::optional<ResolvedValue>
requireReady(const ResolvedValue &resolved,
             evaluator::EvaluatorValuePtr pendingValue,
             llvm::function_ref<void()> emitFailure, ReadyValue &readyValue) {
  switch (resolved.state) {
  case ResolutionState::Pending:
    return ResolvedValue::pending(std::move(pendingValue));
  case ResolutionState::Failure:
    emitFailure();
    return ResolvedValue::failure();
  case ResolutionState::Ready:
    readyValue = materializeReadyValue(resolved, "expected ready value");
    return std::nullopt;
  }
  llvm_unreachable("unknown resolution state");
}

static std::optional<ResolvedValue> requireAllOperandsReady(
    ValueRange operands, evaluator::EvaluatorValuePtr pendingValue,
    llvm::function_ref<ResolvedValue(Value)> evaluateOperand,
    llvm::function_ref<void()> emitFailure,
    SmallVectorImpl<ReadyValue> &readyOperands) {
  readyOperands.clear();
  readyOperands.reserve(operands.size());

  for (auto operand : operands) {
    ReadyValue readyOperand;
    if (auto early = requireReady(evaluateOperand(operand), pendingValue,
                                  emitFailure, readyOperand))
      return *early;
    readyOperands.push_back(std::move(readyOperand));
  }

  return std::nullopt;
}

static ResolvedValue markUnknownAndReturn(evaluator::EvaluatorValuePtr value) {
  value->markUnknown();
  return inspectValue(std::move(value));
}

static bool hasUnknownValue(ArrayRef<ReadyValue> values) {
  for (const auto &value : values)
    if (value.isUnknown())
      return true;
  return false;
}

static ResolvedValue setAttrResult(evaluator::EvaluatorValuePtr resultValue,
                                   Attribute attr) {
  auto *attrValue = cast<evaluator::AttributeValue>(resultValue.get());
  if (failed(attrValue->setAttr(attr)) || failed(attrValue->finalize()))
    return ResolvedValue::failure();
  return inspectValue(std::move(resultValue));
}

class OperationPattern {
public:
  using CreatePartialValueFn =
      llvm::function_ref<FailureOr<evaluator::EvaluatorValuePtr>(Type,
                                                                 Location)>;
  using GetValueHandleFn =
      llvm::function_ref<FailureOr<evaluator::EvaluatorValuePtr>(Value,
                                                                 Location)>;

  explicit OperationPattern(StringRef operationName)
      : operationName(operationName) {}
  virtual ~OperationPattern() = default;

  StringRef getOperationName() const { return operationName; }
  virtual FailureOr<evaluator::EvaluatorValuePtr>
  createPlaceholder(Operation *op, Value value,
                    CreatePartialValueFn createPartialValue,
                    GetValueHandleFn getValueHandle, Location loc) const {
    return createPartialValue(value.getType(), loc);
  }

  virtual ResolvedValue
  evaluate(Operation *op, evaluator::EvaluatorValuePtr resultValue,
           llvm::function_ref<ResolvedValue(Value)> evaluateValue,
           Location loc) const = 0;

private:
  StringRef operationName;
};

template <typename OpT>
class OpPattern : public OperationPattern {
public:
  using OperationPattern::OperationPattern;

  FailureOr<evaluator::EvaluatorValuePtr> createPlaceholder(
      Operation *op, Value value, CreatePartialValueFn createPartialValue,
      GetValueHandleFn getValueHandle, Location loc) const override {
    return createTypedPlaceholder(cast<OpT>(op), value, createPartialValue,
                                  getValueHandle, loc);
  }

protected:
  static OpT getOp(Operation *op) { return cast<OpT>(op); }

  virtual FailureOr<evaluator::EvaluatorValuePtr>
  createTypedPlaceholder(OpT op, Value value,
                         CreatePartialValueFn createPartialValue,
                         GetValueHandleFn getValueHandle, Location loc) const {
    return OperationPattern::createPlaceholder(op, value, createPartialValue,
                                               getValueHandle, loc);
  }
};

class ReadyOperandsPattern : public OperationPattern {
public:
  using OperationPattern::OperationPattern;

  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const final {
    if (resultValue && resultValue->isFullyEvaluated())
      return inspectValue(std::move(resultValue));

    SmallVector<ReadyValue, 4> readyOperands;
    if (auto early = requireAllOperandsReady(
            op->getOperands(), resultValue, evaluateValue,
            [&] {
              op->emitError()
                  << "failed to resolve " << getOperationName() << " operand";
            },
            readyOperands))
      return *early;

    return evaluateReady(op, readyOperands, std::move(resultValue), loc);
  }

private:
  virtual ResolvedValue evaluateReady(Operation *op,
                                      ArrayRef<ReadyValue> operands,
                                      evaluator::EvaluatorValuePtr resultValue,
                                      Location loc) const = 0;
};

template <typename OpT>
class OpReadyOperandsPattern : public ReadyOperandsPattern {
public:
  using ReadyOperandsPattern::ReadyOperandsPattern;

private:
  FailureOr<evaluator::EvaluatorValuePtr>
  createPlaceholder(Operation *op, Value value,
                    CreatePartialValueFn createPartialValue,
                    GetValueHandleFn getValueHandle, Location loc) const final {
    return createTypedPlaceholder(cast<OpT>(op), value, createPartialValue,
                                  getValueHandle, loc);
  }

  ResolvedValue evaluateReady(Operation *op, ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const final {
    return evaluateTyped(cast<OpT>(op), operands, std::move(resultValue), loc);
  }

  virtual FailureOr<evaluator::EvaluatorValuePtr>
  createTypedPlaceholder(OpT op, Value value,
                         CreatePartialValueFn createPartialValue,
                         GetValueHandleFn getValueHandle, Location loc) const {
    return OperationPattern::createPlaceholder(op, value, createPartialValue,
                                               getValueHandle, loc);
  }

  virtual ResolvedValue evaluateTyped(OpT op, ArrayRef<ReadyValue> operands,
                                      evaluator::EvaluatorValuePtr resultValue,
                                      Location loc) const = 0;
};

class IntegerBinaryArithmeticPattern final
    : public OpReadyOperandsPattern<IntegerBinaryArithmeticOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;

private:
  ResolvedValue evaluateTyped(IntegerBinaryArithmeticOp op,
                              ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 2 && "expected binary arithmetic operands");
    if (hasUnknownValue(operands))
      return markUnknownAndReturn(std::move(resultValue));

    circt::om::IntegerAttr lhs = operands[0].getIntegerAttr(
        "expected om::IntegerAttr for IntegerBinaryArithmeticOp lhs");
    circt::om::IntegerAttr rhs = operands[1].getIntegerAttr(
        "expected om::IntegerAttr for IntegerBinaryArithmeticOp rhs");

    APSInt lhsVal = lhs.getValue().getAPSInt();
    APSInt rhsVal = rhs.getValue().getAPSInt();
    if (lhsVal.getBitWidth() > rhsVal.getBitWidth())
      rhsVal = rhsVal.extend(lhsVal.getBitWidth());
    else if (rhsVal.getBitWidth() > lhsVal.getBitWidth())
      lhsVal = lhsVal.extend(rhsVal.getBitWidth());

    FailureOr<APSInt> result = op.evaluateIntegerOperation(lhsVal, rhsVal);
    if (failed(result))
      return (op->emitError("failed to evaluate integer operation"),
              ResolvedValue::failure());

    MLIRContext *ctx = op.getContext();
    auto resultAttr = circt::om::IntegerAttr::get(
        ctx, mlir::IntegerAttr::get(ctx, result.value()));
    return setAttrResult(std::move(resultValue), resultAttr);
  }
};

class ListCreatePattern final : public OpReadyOperandsPattern<ListCreateOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;

private:
  ResolvedValue evaluateTyped(ListCreateOp op, ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    SmallVector<evaluator::EvaluatorValuePtr> values;
    values.reserve(operands.size());
    for (const auto &operand : operands)
      values.push_back(operand.handle);

    cast<evaluator::ListValue>(resultValue.get())
        ->setElements(std::move(values));
    if (hasUnknownValue(operands))
      resultValue->markUnknown();
    return inspectValue(std::move(resultValue));
  }
};

class ListConcatPattern final : public OpReadyOperandsPattern<ListConcatOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;

private:
  ResolvedValue evaluateTyped(ListConcatOp op, ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    SmallVector<evaluator::EvaluatorValuePtr> values;
    for (const auto &operand : operands) {
      auto *subListValue =
          operand.getListValue("expected list value for list_concat operand");
      llvm::append_range(values, subListValue->getElements());
    }

    cast<evaluator::ListValue>(resultValue.get())
        ->setElements(std::move(values));
    if (hasUnknownValue(operands))
      resultValue->markUnknown();
    return inspectValue(std::move(resultValue));
  }
};

class StringConcatPattern final
    : public OpReadyOperandsPattern<StringConcatOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;

private:
  ResolvedValue evaluateTyped(StringConcatOp op, ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    if (hasUnknownValue(operands))
      return markUnknownAndReturn(std::move(resultValue));

    std::string result;
    for (const auto &operand : operands)
      result +=
          operand
              .getStringAttr("expected StringAttr for StringConcatOp operand")
              .getValue()
              .str();

    auto resultStr = StringAttr::get(result, op.getResult().getType());
    return setAttrResult(std::move(resultValue), resultStr);
  }
};

class BinaryEqualityPattern final
    : public OpReadyOperandsPattern<BinaryEqualityOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;

private:
  ResolvedValue evaluateTyped(BinaryEqualityOp op,
                              ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 2 && "expected binary equality operands");
    if (hasUnknownValue(operands))
      return markUnknownAndReturn(std::move(resultValue));

    mlir::Attribute lhs = operands[0].getAttr(
        "expected attribute value for BinaryEqualityOp lhs");
    mlir::Attribute rhs = operands[1].getAttr(
        "expected attribute value for BinaryEqualityOp rhs");

    FailureOr<mlir::Attribute> result = op.evaluateBinaryEquality(lhs, rhs);
    if (failed(result))
      return (op->emitError("failed to evaluate binary equality operation"),
              ResolvedValue::failure());
    return setAttrResult(std::move(resultValue), *result);
  }
};

class FrozenBasePathCreatePattern final
    : public OpReadyOperandsPattern<FrozenBasePathCreateOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;
  FailureOr<evaluator::EvaluatorValuePtr>
  createTypedPlaceholder(FrozenBasePathCreateOp op, Value value,
                         CreatePartialValueFn createPartialValue,
                         GetValueHandleFn getValueHandle,
                         Location loc) const override {
    return success(
        std::make_shared<evaluator::BasePathValue>(op.getPathAttr(), loc));
  }

private:
  ResolvedValue evaluateTyped(FrozenBasePathCreateOp op,
                              ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 1 &&
           "expected one operand for frozenbasepath_create");
    if (operands.front().isUnknown())
      return markUnknownAndReturn(std::move(resultValue));

    auto *basePathValue = operands.front().getBasePathValue(
        "resolved frozenbasepath_create operand is not a base path");
    cast<evaluator::BasePathValue>(resultValue.get())
        ->setBasepath(*basePathValue);
    return inspectValue(std::move(resultValue));
  }
};

class FrozenPathCreatePattern final
    : public OpReadyOperandsPattern<FrozenPathCreateOp> {
public:
  using OpReadyOperandsPattern::OpReadyOperandsPattern;
  FailureOr<evaluator::EvaluatorValuePtr>
  createTypedPlaceholder(FrozenPathCreateOp pathOp, Value value,
                         CreatePartialValueFn createPartialValue,
                         GetValueHandleFn getValueHandle,
                         Location loc) const override {
    return success(std::make_shared<evaluator::PathValue>(
        pathOp.getTargetKindAttr(), pathOp.getPathAttr(),
        pathOp.getModuleAttr(), pathOp.getRefAttr(), pathOp.getFieldAttr(),
        loc));
  }

private:
  ResolvedValue evaluateTyped(FrozenPathCreateOp op,
                              ArrayRef<ReadyValue> operands,
                              evaluator::EvaluatorValuePtr resultValue,
                              Location loc) const override {
    assert(operands.size() == 1 &&
           "expected one operand for frozenpath_create");
    if (operands.front().isUnknown())
      return markUnknownAndReturn(std::move(resultValue));

    auto *basePathValue = operands.front().getBasePathValue(
        "resolved frozenpath_create operand is not a base path");
    cast<evaluator::PathValue>(resultValue.get())->setBasepath(*basePathValue);
    return inspectValue(std::move(resultValue));
  }
};

class ConstantPattern final : public OpPattern<ConstantOp> {
public:
  using OpPattern::OpPattern;
  FailureOr<evaluator::EvaluatorValuePtr> createTypedPlaceholder(
      ConstantOp op, Value value, CreatePartialValueFn createPartialValue,
      GetValueHandleFn getValueHandle, Location loc) const override {
    return success(
        circt::om::evaluator::AttributeValue::get(op.getValue(), loc));
  }

  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const override {
    if (resultValue && resultValue->isFullyEvaluated())
      return inspectValue(std::move(resultValue));
    return inspectValue(
        circt::om::evaluator::AttributeValue::get(getOp(op).getValue(), loc));
  }
};

class AnyCastPattern final : public OpPattern<AnyCastOp> {
public:
  using OpPattern::OpPattern;
  FailureOr<evaluator::EvaluatorValuePtr> createTypedPlaceholder(
      AnyCastOp op, Value value, CreatePartialValueFn createPartialValue,
      GetValueHandleFn getValueHandle, Location loc) const override {
    return getValueHandle(op.getInput(), loc);
  }

  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const override {
    if (resultValue && resultValue->isFullyEvaluated())
      return inspectValue(std::move(resultValue));
    return evaluateValue(getOp(op).getInput());
  }
};

class FrozenEmptyPathPattern final : public OpPattern<FrozenEmptyPathOp> {
public:
  using OpPattern::OpPattern;
  FailureOr<evaluator::EvaluatorValuePtr>
  createTypedPlaceholder(FrozenEmptyPathOp op, Value value,
                         CreatePartialValueFn createPartialValue,
                         GetValueHandleFn getValueHandle,
                         Location loc) const override {
    return success(std::make_shared<evaluator::PathValue>(
        evaluator::PathValue::getEmptyPath(loc)));
  }

  ResolvedValue evaluate(Operation *op,
                         evaluator::EvaluatorValuePtr resultValue,
                         llvm::function_ref<ResolvedValue(Value)> evaluateValue,
                         Location loc) const override {
    return inspectValue(std::move(resultValue));
  }
};

class OperationPatternRegistry {
public:
  OperationPatternRegistry() {
    addPattern(
        ConstantOp::getOperationName(),
        std::make_unique<ConstantPattern>(ConstantOp::getOperationName()));
    addPattern(AnyCastOp::getOperationName(),
               std::make_unique<AnyCastPattern>(AnyCastOp::getOperationName()));
    addPattern(FrozenEmptyPathOp::getOperationName(),
               std::make_unique<FrozenEmptyPathPattern>(
                   FrozenEmptyPathOp::getOperationName()));
    addPattern(IntegerAddOp::getOperationName(),
               std::make_unique<IntegerBinaryArithmeticPattern>(
                   IntegerAddOp::getOperationName()));
    addPattern(IntegerMulOp::getOperationName(),
               std::make_unique<IntegerBinaryArithmeticPattern>(
                   IntegerMulOp::getOperationName()));
    addPattern(IntegerShrOp::getOperationName(),
               std::make_unique<IntegerBinaryArithmeticPattern>(
                   IntegerShrOp::getOperationName()));
    addPattern(IntegerShlOp::getOperationName(),
               std::make_unique<IntegerBinaryArithmeticPattern>(
                   IntegerShlOp::getOperationName()));
    addPattern(
        ListCreateOp::getOperationName(),
        std::make_unique<ListCreatePattern>(ListCreateOp::getOperationName()));
    addPattern(
        ListConcatOp::getOperationName(),
        std::make_unique<ListConcatPattern>(ListConcatOp::getOperationName()));
    addPattern(StringConcatOp::getOperationName(),
               std::make_unique<StringConcatPattern>(
                   StringConcatOp::getOperationName()));
    addPattern(
        PropEqOp::getOperationName(),
        std::make_unique<BinaryEqualityPattern>(PropEqOp::getOperationName()));
    addPattern(FrozenBasePathCreateOp::getOperationName(),
               std::make_unique<FrozenBasePathCreatePattern>(
                   FrozenBasePathCreateOp::getOperationName()));
    addPattern(FrozenPathCreateOp::getOperationName(),
               std::make_unique<FrozenPathCreatePattern>(
                   FrozenPathCreateOp::getOperationName()));
  }

  const OperationPattern *lookup(Operation *op) const {
    auto it = patternsByOpName.find(op->getName().getStringRef());
    return it == patternsByOpName.end() ? nullptr : it->second;
  }

private:
  void addPattern(StringRef opName, std::unique_ptr<OperationPattern> pattern) {
    const OperationPattern *patternPtr = pattern.get();
    patterns.push_back(std::move(pattern));
    patternsByOpName[opName] = patternPtr;
  }

  SmallVector<std::unique_ptr<OperationPattern>> patterns;
  llvm::StringMap<const OperationPattern *> patternsByOpName;
};

static const OperationPatternRegistry &getOperationPatternRegistry() {
  static const OperationPatternRegistry registry;
  return registry;
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
      .Case([&](FrozenBasePathType type) {
        return success(
            std::make_shared<evaluator::BasePathValue>(type.getContext()));
      })
      .Case([&](FrozenPathType type) {
        return success(std::make_shared<evaluator::PathValue>(
            evaluator::PathValue::getEmptyPath(loc)));
      })
      .Default([&](Type type) {
        return success(evaluator::AttributeValue::get(type, LocationAttr(loc)));
      });
}

FailureOr<evaluator::EvaluatorValuePtr>
circt::om::Evaluator::createPlaceholderValue(Operation *op, Value value,
                                             ActualParameters actualParams,
                                             Location loc) {
  using namespace circt::om::evaluator;

  if (auto *pattern = getOperationPatternRegistry().lookup(op))
    return pattern->createPlaceholder(
        op, value,
        [&](Type type, Location placeholderLoc) {
          return getPartiallyEvaluatedValue(type, placeholderLoc);
        },
        [&](Value aliasedValue, Location placeholderLoc) {
          return getOrCreateValue(aliasedValue, actualParams, placeholderLoc);
        },
        loc);

  return TypeSwitch<Operation *, FailureOr<evaluator::EvaluatorValuePtr>>(op)
      .Case<ObjectFieldOp>([&](auto op) {
        return success(std::make_shared<ReferenceValue>(value.getType(), loc));
      })
      .Case<ObjectOp>([&](auto op) {
        return getPartiallyEvaluatedValue(op.getType(), op.getLoc());
      })
      .Case<UnknownValueOp>(
          [&](auto op) { return createUnknownValue(op.getType(), loc); })
      .Default([&](Operation *op) {
        auto error = op->emitError("unable to evaluate value");
        error.attachNote() << "value: " << value;
        return error;
      });
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
            return createPlaceholderValue(result.getDefiningOp(), value,
                                          actualParams, loc);
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
    auto result = evaluateValue(value, actualParams, fieldLoc);
    if (result.state == ResolutionState::Failure)
      return failure();

    fields[cast<StringAttr>(name)] = result.value;
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

    if (result.state == ResolutionState::Failure)
      return failure();

    // It's possible that the value is not fully evaluated.
    if (result.state == ResolutionState::Pending)
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

ResolvedValue circt::om::Evaluator::evaluateValue(Value value,
                                                  ActualParameters actualParams,
                                                  Location loc) {
  auto evaluatorValue = getOrCreateValue(value, actualParams, loc);
  if (failed(evaluatorValue))
    return ResolvedValue::failure();

  return llvm::TypeSwitch<Value, ResolvedValue>(value)
      .Case([&](BlockArgument arg) {
        return evaluateParameter(arg, actualParams, loc);
      })
      .Case<OpResult>([&](OpResult result) {
        if (auto *pattern =
                getOperationPatternRegistry().lookup(result.getDefiningOp()))
          return pattern->evaluate(
              result.getDefiningOp(), evaluatorValue.value(),
              [&](Value nestedValue) {
                return evaluateValue(nestedValue, actualParams, loc);
              },
              loc);

        if (evaluatorValue.value()->isFullyEvaluated())
          return inspectValue(evaluatorValue.value());

        return TypeSwitch<Operation *, ResolvedValue>(result.getDefiningOp())
            .Case([&](ObjectOp op) {
              return evaluateObjectInstance(op, actualParams);
            })
            .Case([&](ObjectFieldOp op) {
              return evaluateObjectField(op, actualParams, loc);
            })
            .Case<UnknownValueOp>([&](UnknownValueOp op) {
              return evaluateUnknownValue(op, loc);
            })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return ResolvedValue::failure();
            });
      });
}

/// Evaluator dispatch function for parameters.
ResolvedValue circt::om::Evaluator::evaluateParameter(
    BlockArgument formalParam, ActualParameters actualParams, Location loc) {
  auto val = (*actualParams)[formalParam.getArgNumber()];
  val->setLoc(loc);
  return inspectValue(val);
}

/// Evaluator dispatch function for property assertions.
LogicalResult
circt::om::Evaluator::evaluatePropertyAssert(PropertyAssertOp op,
                                             ActualParameters actualParams) {
  auto loc = op.getLoc();
  ReadyValue readyCond;
  if (auto early = requireReady(
          evaluateValue(op.getCondition(), actualParams, loc), nullptr,
          [&] {
            op.emitError("failed to resolve property assertion condition");
          },
          readyCond))
    return early->state == ResolutionState::Pending ? success() : failure();

  if (readyCond.isUnknown())
    return success();

  auto condAttr =
      readyCond.getAttr("expected attribute value for property assertion");

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
      if (evaluatedInput.state == ResolutionState::Failure)
        return failure();
      if (evaluatedInput.value)
        inputValue = evaluatedInput.value;
    }
    parameters->push_back(inputValue);
  }

  actualParametersBuffers.push_back(std::move(parameters));
  return actualParametersBuffers.back().get();
}

/// Evaluator dispatch function for Object instances.
ResolvedValue
circt::om::Evaluator::evaluateObjectInstance(ObjectOp op,
                                             ActualParameters actualParams) {
  auto loc = op.getLoc();
  auto key = ObjectKey{op, actualParams};
  if (isFullyEvaluated(key))
    return inspectValue(getOrCreateValue(op, actualParams, loc).value());
  if (!activeObjectInstances.insert(key).second)
    return inspectValue(getOrCreateValue(op, actualParams, loc).value());
  auto clearActiveObject =
      llvm::scope_exit([&] { activeObjectInstances.erase(key); });

  auto params =
      createParametersFromOperands(op.getOperands(), actualParams, loc);
  if (failed(params))
    return ResolvedValue::failure();
  auto result = evaluateObjectInstance(op.getClassNameAttr(), params.value(),
                                       loc, {op, actualParams});
  if (failed(result))
    return ResolvedValue::failure();
  return inspectValue(result.value());
}

/// Evaluator dispatch function for Object fields.
ResolvedValue circt::om::Evaluator::evaluateObjectField(
    ObjectFieldOp op, ActualParameters actualParams, Location loc) {
  auto objectFieldValue = getOrCreateValue(op, actualParams, loc).value();

  auto setUnknownFieldValue = [&]() -> ResolvedValue {
    auto unknownField = createUnknownValue(op.getResult().getType(), loc);
    if (failed(unknownField))
      return ResolvedValue::failure();

    if (auto *ref =
            llvm::dyn_cast<evaluator::ReferenceValue>(objectFieldValue.get()))
      ref->setValue(unknownField.value());

    objectFieldValue->markUnknown();
    return ResolvedValue::ready(objectFieldValue);
  };

  ReadyValue readyObject;
  if (auto early = requireReady(
          evaluateValue(op.getObject(), actualParams, loc), objectFieldValue,
          [&] { op.emitError("failed to resolve object field base"); },
          readyObject))
    return *early;

  if (readyObject.isUnknown()) {
    auto unknownField = setUnknownFieldValue();
    if (unknownField.state != ResolutionState::Ready)
      return ResolvedValue::failure();
    return inspectValue(unknownField.value);
  }

  auto *currentObject =
      readyObject.getObjectValue("resolved object field base is not an object");

  // Iteratively access nested fields through the path until we reach the final
  // field in the path.
  evaluator::EvaluatorValuePtr finalField;
  auto fieldPath = op.getFieldPath().getAsRange<FlatSymbolRefAttr>();
  for (auto it = fieldPath.begin(), end = fieldPath.end(); it != end; ++it) {
    auto field = *it;
    // `currentObject` might no be fully evaluated.
    if (!currentObject->getFields().contains(field.getAttr()))
      return ResolvedValue::pending(objectFieldValue);

    auto currentField = currentObject->getField(field.getAttr());
    finalField = currentField.value();
    if (std::next(it) == end)
      continue;

    ReadyValue nextObject;
    if (auto early = requireReady(
            inspectValue(finalField), objectFieldValue,
            [&] {
              op.emitError("failed to resolve nested object field "
                           "path");
            },
            nextObject))
      return *early;
    if (nextObject.isUnknown()) {
      auto unknownField = setUnknownFieldValue();
      if (unknownField.state != ResolutionState::Ready)
        return ResolvedValue::failure();
      return inspectValue(unknownField.value);
    }

    currentObject = nextObject.getObjectValue(
        "resolved nested object field path is not an object");
  }

  // Update the reference.
  llvm::cast<evaluator::ReferenceValue>(objectFieldValue.get())
      ->setValue(finalField);

  // Return the field being accessed.
  return inspectValue(objectFieldValue);
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
ResolvedValue circt::om::Evaluator::evaluateUnknownValue(UnknownValueOp op,
                                                         Location loc) {
  auto result = createUnknownValue(op.getType(), loc);
  if (failed(result))
    return ResolvedValue::failure();
  return inspectValue(result.value());
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
