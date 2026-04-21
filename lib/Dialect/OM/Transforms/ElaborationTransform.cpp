//===- ElaborationTransform.cpp - OM elaboration transform ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Transforms/ElaborationTransform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace circt;
using namespace circt::om;

namespace {

class ElaborationTransform {
public:
  ElaborationTransform(ClassOp wrapperClass, SymbolTable &symbols)
      : wrapperClass(wrapperClass), symbols(symbols),
        builder(wrapperClass.getContext()) {}

  LogicalResult run() {
    for (Value field : wrapperClass.getFieldsOp().getFields()) {
      auto objectOp = field.getDefiningOp<ObjectOp>();
      if (!objectOp)
        continue;
      if (failed(prepareObject(objectOp)))
        return failure();
    }
    if (failed(rewriteWrapper()))
      return failure();
    return greedilySimplifyWrapper();
  }

private:
  static bool wouldCreateCyclicReplacement(ObjectFieldOp op,
                                           Value replacement) {
    if (replacement == op.getResult())
      return true;
    Operation *replacementOp = replacement.getDefiningOp();
    return replacementOp &&
           llvm::is_contained(replacementOp->getOperands(), op.getResult());
  }

  template <typename OpTy>
  struct AttrFoldConversionPattern : OpConversionPattern<OpTy> {
    AttrFoldConversionPattern(MLIRContext *context)
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
      auto typedAttr = dyn_cast_if_present<TypedAttr>(
          dyn_cast<Attribute>(foldResults.front()));
      if (!typedAttr)
        return failure();
      rewriter.replaceOpWithNewOp<ConstantOp>(op, typedAttr);
      return success();
    }
  };

  template <typename OpTy>
  struct AttrFoldRewritePattern : OpRewritePattern<OpTy> {
    AttrFoldRewritePattern(MLIRContext *context)
        : OpRewritePattern<OpTy>(context) {}

    LogicalResult matchAndRewrite(OpTy op,
                                  PatternRewriter &rewriter) const override {
      bool hasUnknownOperand = false;
      SmallVector<Attribute> operandAttrs;
      operandAttrs.reserve(op->getNumOperands());
      for (Value operand : op->getOperands()) {
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
      auto typedAttr = dyn_cast_if_present<TypedAttr>(
          dyn_cast<Attribute>(foldResults.front()));
      if (!typedAttr)
        return failure();
      rewriter.replaceOpWithNewOp<ConstantOp>(op, typedAttr);
      return success();
    }
  };

  struct ObjectOpConversionPattern : OpConversionPattern<ObjectOp> {
    ObjectOpConversionPattern(MLIRContext *context, ElaborationTransform &state)
        : OpConversionPattern<ObjectOp>(context), state(state) {}

    LogicalResult
    matchAndRewrite(ObjectOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto converted = state.convertObject(op, rewriter);
      if (failed(converted))
        return failure();
      rewriter.replaceOp(op, converted.value());
      return success();
    }

    ElaborationTransform &state;
  };

  struct ObjectFieldConversionPattern : OpConversionPattern<ObjectFieldOp> {
    ObjectFieldConversionPattern(MLIRContext *context,
                                 ElaborationTransform &state)
        : OpConversionPattern<ObjectFieldOp>(context), state(state) {}

    LogicalResult
    matchAndRewrite(ObjectFieldOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      auto simplified = state.simplifyObjectField(
          adaptor.getObject(), op.getFieldAttr(), op.getType(), rewriter);
      if (failed(simplified) || !*simplified)
        return failure();
      if (wouldCreateCyclicReplacement(op, **simplified))
        return failure();
      rewriter.replaceOp(op, **simplified);
      return success();
    }

    ElaborationTransform &state;
  };

  struct ObjectFieldRewritePattern : OpRewritePattern<ObjectFieldOp> {
    ObjectFieldRewritePattern(MLIRContext *context, ElaborationTransform &state)
        : OpRewritePattern<ObjectFieldOp>(context), state(state) {}

    LogicalResult matchAndRewrite(ObjectFieldOp op,
                                  PatternRewriter &rewriter) const override {
      auto simplified = state.simplifyObjectField(
          op.getObject(), op.getFieldAttr(), op.getType(), rewriter);
      if (failed(simplified) || !*simplified)
        return failure();
      if (wouldCreateCyclicReplacement(op, **simplified))
        return failure();
      rewriter.replaceOp(op, **simplified);
      return success();
    }

    ElaborationTransform &state;
  };

  FailureOr<ClassLike> lookupClassLike(StringAttr className) {
    if (auto classLike = symbols.lookup<ClassLike>(className))
      return classLike;
    return wrapperClass.emitError("unknown class name ") << className;
  }

  FailureOr<ClassFieldsOp> findWrapperFieldsOp() {
    for (Block &block : wrapperClass.getBody())
      if (!block.empty())
        if (auto fieldsOp = dyn_cast<ClassFieldsOp>(&block.back()))
          return fieldsOp;
    return wrapperClass.emitError("wrapper class is missing om.class.fields");
  }

  Value createUnknownScratchValue(Type type, Location loc, OpBuilder &builder) {
    return UnknownValueOp::create(builder, loc, type).getResult();
  }

  LogicalResult prepareObject(ObjectOp objectOp) {
    if (!preparedObjects.insert(objectOp.getOperation()).second)
      return success();

    auto classLike = lookupClassLike(objectOp.getClassNameAttr());
    if (failed(classLike))
      return failure();
    if (isa<ClassExternOp>(*classLike))
      return success();

    auto classOp = dyn_cast<ClassOp>(classLike.value().getOperation());
    assert(classOp && "non-external class should be a ClassOp");

    auto fieldsOp = findWrapperFieldsOp();
    if (failed(fieldsOp))
      return failure();
    builder.setInsertionPoint(*fieldsOp);
    SmallVector<Value> fieldPlaceholders;
    fieldPlaceholders.reserve(classOp.getFieldsOp().getFields().size());
    for (Value field : classOp.getFieldsOp().getFields())
      fieldPlaceholders.push_back(createUnknownScratchValue(
          field.getType(), objectOp.getLoc(), builder));

    auto elaborated = ElaboratedObjectOp::create(builder, objectOp.getLoc(),
                                                 *classLike, fieldPlaceholders);
    elaboratedObjects[objectOp.getOperation()] = elaborated.getResult();

    Region clonedRegion;
    IRMapping mapper;
    for (auto [formal, actual] :
         llvm::zip(classOp.getBodyBlock()->getArguments(),
                   objectOp.getActualParams()))
      mapper.map(formal, actual);
    classOp.getBody().cloneInto(&clonedRegion, mapper);

    Block *clonedBlock = &clonedRegion.front();
    auto clonedFields = cast<ClassFieldsOp>(clonedBlock->getTerminator());
    SmallVector<Value> clonedFieldValues;
    clonedFieldValues.reserve(classOp.getFieldsOp().getFields().size());
    llvm::append_range(clonedFieldValues, clonedFields.getFields());
    IRRewriter rewriter(wrapperClass.getContext());
    rewriter.inlineBlockBefore(clonedBlock, fieldsOp->getOperation());
    elaborated->setOperands(clonedFieldValues);
    clonedFields.erase();

    SmallVector<ObjectOp> nestedObjects;
    wrapperClass.walk([&](ObjectOp nestedObject) {
      if (nestedObject != objectOp)
        nestedObjects.push_back(nestedObject);
    });
    for (auto nestedObject : nestedObjects)
      if (failed(prepareObject(nestedObject)))
        return failure();
    return success();
  }

  FailureOr<Value> convertObject(ObjectOp objectOp,
                                 ConversionPatternRewriter &rewriter) {
    auto classLike = lookupClassLike(objectOp.getClassNameAttr());
    if (failed(classLike))
      return failure();

    if (isa<ClassExternOp>(*classLike)) {
      rewriter.setInsertionPoint(wrapperClass.getFieldsOp());
      return createUnknownScratchValue(objectOp.getType(), objectOp.getLoc(),
                                       rewriter);
    }

    auto classOp = dyn_cast<ClassOp>(classLike.value().getOperation());
    assert(classOp && "non-external class should be a ClassOp");
    auto it = elaboratedObjects.find(objectOp.getOperation());
    if (it == elaboratedObjects.end())
      return objectOp.emitError(
          "missing elaborated object for converted object");
    return it->second;
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

    auto classLike = lookupClassLike(elaboratedObject.getClassNameAttr());
    if (failed(classLike))
      return failure();

    auto fieldNames = classLike->getFieldNames();
    auto fieldIt = llvm::find(fieldNames, field.getAttr());
    if (fieldIt == fieldNames.end())
      return elaboratedObject.emitError("field ") << field << " does not exist";
    return std::optional<Value>(
        elaboratedObject
            .getFieldValues()[std::distance(fieldNames.begin(), fieldIt)]);
  }

  LogicalResult rewriteWrapper() {
    ConversionTarget target(*wrapperClass.getContext());
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

    RewritePatternSet conversionPatterns(wrapperClass.getContext());
    conversionPatterns
        .add<ObjectOpConversionPattern, ObjectFieldConversionPattern>(
            wrapperClass.getContext(), *this);
    conversionPatterns.add<AttrFoldConversionPattern<IntegerAddOp>,
                           AttrFoldConversionPattern<IntegerMulOp>,
                           AttrFoldConversionPattern<IntegerShrOp>,
                           AttrFoldConversionPattern<IntegerShlOp>,
                           AttrFoldConversionPattern<StringConcatOp>,
                           AttrFoldConversionPattern<PropEqOp>>(
        wrapperClass.getContext());
    return applyFullConversion(wrapperClass, target,
                               std::move(conversionPatterns));
  }

  LogicalResult greedilySimplifyWrapper() {
    RewritePatternSet patterns(wrapperClass.getContext());
    patterns.add<ObjectFieldRewritePattern>(wrapperClass.getContext(), *this);
    patterns.add<AttrFoldRewritePattern<IntegerAddOp>,
                 AttrFoldRewritePattern<IntegerMulOp>,
                 AttrFoldRewritePattern<IntegerShrOp>,
                 AttrFoldRewritePattern<IntegerShlOp>,
                 AttrFoldRewritePattern<StringConcatOp>,
                 AttrFoldRewritePattern<PropEqOp>>(wrapperClass.getContext());
    return applyPatternsGreedily(wrapperClass, std::move(patterns));
  }

  ClassOp wrapperClass;
  SymbolTable &symbols;
  OpBuilder builder;
  llvm::SmallDenseSet<Operation *, 16> preparedObjects;
  DenseMap<Operation *, Value> elaboratedObjects;
};

} // namespace

LogicalResult circt::om::applyElaborationTransform(ClassOp wrapperClass,
                                                   SymbolTable &symbols) {
  return ElaborationTransform(wrapperClass, symbols).run();
}
