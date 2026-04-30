//===- ElaborateObject.cpp - OM elaboration pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

namespace circt {
namespace om {
#define GEN_PASS_DEF_ELABORATEOBJECT
#include "circt/Dialect/OM/OMPasses.h.inc"
} // namespace om
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace om;

namespace {
using FieldIndex = DenseMap<std::pair<StringAttr, StringAttr>, unsigned>;

// Pattern to inline ObjectOp and replace with ElaboratedObjectOp
struct ObjectOpInliningPattern : public OpRewritePattern<ObjectOp> {
  ObjectOpInliningPattern(MLIRContext *context, SymbolTable &symbols)
      : OpRewritePattern<ObjectOp>(context), symbols(symbols) {}

  LogicalResult matchAndRewrite(ObjectOp objOp,
                                PatternRewriter &rewriter) const override {
    auto classLike = symbols.lookup<ClassLike>(objOp.getClassNameAttr());
    assert(classLike);

    // External classes become unknown
    if (isa<ClassExternOp>(classLike)) {
      rewriter.replaceOpWithNewOp<UnknownValueOp>(objOp, objOp.getType());
      return success();
    }

    auto classOp = cast<ClassOp>(classLike);

    // Map formal parameters to actual parameters
    IRMapping mapper;
    for (auto [formal, actual] : llvm::zip(
             classOp.getBodyBlock()->getArguments(), objOp.getActualParams()))
      mapper.map(formal, actual);

    // Clone the class body into a temporary region
    Region clonedRegion;
    classOp.getBody().cloneInto(&clonedRegion, mapper);
    Block *clonedBlock = &clonedRegion.front();

    // Get field values from the terminator before inlining
    auto clonedFields = cast<ClassFieldsOp>(clonedBlock->getTerminator());
    SmallVector<Value> fieldValues(clonedFields.getFields());

    // Erase the terminator and inline.
    rewriter.eraseOp(clonedFields);
    rewriter.inlineBlockBefore(clonedBlock, objOp);

    // Replace the ObjectOp with an ElaboratedObjectOp
    rewriter.replaceOpWithNewOp<ElaboratedObjectOp>(objOp, classLike,
                                                    fieldValues);

    return success();
  }

  SymbolTable &symbols;
};

struct ObjectFieldOpConversionPattern : OpRewritePattern<ObjectFieldOp> {
  ObjectFieldOpConversionPattern(MLIRContext *context,
                                 const SymbolTable &symbols,
                                 const FieldIndex &fieldIndexes)
      : OpRewritePattern<ObjectFieldOp>(context), symbols(symbols),
        fieldIndexes(fieldIndexes) {}

  LogicalResult matchAndRewrite(ObjectFieldOp op,
                                PatternRewriter &rewriter) const override {
    // Only fold if the object is an ElaboratedObjectOp
    auto elaboratedOp = op.getObject().getDefiningOp<ElaboratedObjectOp>();
    if (!elaboratedOp)
      return failure();

    // Look up the class to get field names
    auto classLike = symbols.lookup<ClassLike>(elaboratedOp.getClassNameAttr());
    assert(classLike);

    auto index =
        fieldIndexes.at({classLike.getSymNameAttr(), op.getFieldAttr()});
    auto result = elaboratedOp.getFieldValues()[index];
    if (op.getResult() == result)
      return rewriter.notifyMatchFailure(op.getLoc(), "found cycle");

    // Replace with the corresponding field value from the elaborated object
    rewriter.replaceOp(op, result);
    return success();
  }

  const SymbolTable &symbols;
  const FieldIndex &fieldIndexes;
};

// Propagate UnknownValueOp through Pure OM operations
struct UnknownPropagationPattern : RewritePattern {
  UnknownPropagationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<OMDialect>(op->getDialect()) || !isPure(op))
      return failure();

    // Check if any operand is an UnknownValueOp
    // TODO: This is directly port of the existing Evaluator sematics, but it
    // causes inconsistent evaluation for operations that can reason about
    // "known" value "and(0, unknown) -> 0".
    if (!llvm::any_of(op->getOperands(), [](Value operand) {
          return operand.getDefiningOp<UnknownValueOp>();
        }))
      return failure();

    // Replace with UnknownValueOp for each result
    SmallVector<Value> unknowns;
    for (Type resultType : op->getResultTypes())
      unknowns.push_back(
          UnknownValueOp::create(rewriter, op->getLoc(), resultType));

    rewriter.replaceOp(op, unknowns);
    return success();
  }
};

struct ElaborateObjectPass
    : public circt::om::impl::ElaborateObjectBase<ElaborateObjectPass> {

  LogicalResult elaborateClass(ClassOp classOp, SymbolTable &symbols,
                               FieldIndex &fieldIndexes) {
    // Elaboration can be performed by inlining all ObjectOps and constant folds
    // using greedy pattern rewriter.
    // NOTE: Conversion framework didn't work well with inlining because
    //       inlining pattern needs to be applied recursively.
    RewritePatternSet patterns(classOp.getContext());
    patterns.add<ObjectOpInliningPattern>(classOp.getContext(), symbols);
    patterns.add<ObjectFieldOpConversionPattern>(classOp.getContext(), symbols,
                                                 fieldIndexes);
    patterns.add<UnknownPropagationPattern>(classOp.getContext());
    GreedyRewriteConfig config;
    // Disable iteration limit.
    config.setMaxIterations(GreedyRewriteConfig::kNoLimit);
    if (failed(applyPatternsGreedily(classOp, std::move(patterns), config)))
      return failure();

    // Check that the class is fully elaborated(= serializable)
    classOp.walk([](Operation *op) {

    });

    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto &symbols = getAnalysis<SymbolTable>();
    FieldIndex fieldIndexes;
    for (auto classOp : module.getOps<ClassLike>()) {
      auto name = classOp.getSymNameAttr();
      for (auto [idx, fieldName] :
           llvm::enumerate(classOp.getFieldNames().getAsRange<StringAttr>()))
        fieldIndexes[{name, fieldName}] = idx;
    }

    // Test mode: elaborate all nullary classes
    if (test) {
      for (auto classOp : module.getOps<ClassOp>()) {
        if (classOp.getBodyBlock()->getNumArguments() == 0)
          if (failed(elaborateClass(classOp, symbols, fieldIndexes)))
            return signalPassFailure();
      }
      return;
    }

    // Target class must be specified
    if (targetClass.empty()) {
      module.emitError("om-elaborate-object requires --target-class option ");
      return signalPassFailure();
    }

    // Find the target class
    auto classOp = symbols.lookup<ClassOp>(targetClass);
    if (!classOp) {
      module.emitError("om-elaborate-object could not find class ")
          << targetClass;
      return signalPassFailure();
    }

    // Only accept classes with zero inputs
    if (classOp.getBodyBlock()->getNumArguments() != 0) {
      classOp.emitError(
          "om-elaborate-object only accepts zero-input classes, but ")
          << targetClass << " has " << classOp.getBodyBlock()->getNumArguments()
          << " inputs";
      return signalPassFailure();
    }

    if (failed(elaborateClass(classOp, symbols, fieldIndexes)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> circt::om::createElaborateObjectPass() {
  return std::make_unique<ElaborateObjectPass>();
}
