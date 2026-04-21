//===- ElaborateObject.cpp - OM elaboration pass --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/OM/Transforms/ElaborationTransform.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

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

static FailureOr<ClassOp>
createTestWrapper(ModuleOp module, SymbolTable &symbols, ClassOp targetClass) {
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToEnd(module.getBody());

  std::string wrapperName =
      ("__om_elaborated_" + targetClass.getSymName()).str();
  unsigned suffix = 0;
  while (symbols.lookup<ClassLike>(builder.getStringAttr(wrapperName)))
    wrapperName = ("__om_elaborated_" + targetClass.getSymName() + "_" +
                   std::to_string(++suffix))
                      .str();

  auto rootType =
      ClassType::get(module.getContext(), FlatSymbolRefAttr::get(targetClass));
  auto wrapperClass = ClassOp::create(
      builder, targetClass.getLoc(), builder.getStringAttr(wrapperName),
      builder.getStrArrayAttr({}),
      builder.getArrayAttr({builder.getStringAttr("root")}),
      builder.getDictionaryAttr(
          {builder.getNamedAttr("root", TypeAttr::get(rootType))}));

  Block *body = &wrapperClass.getRegion().emplaceBlock();
  builder.setInsertionPointToEnd(body);
  auto rootObject = ObjectOp::create(builder, targetClass.getLoc(), targetClass,
                                     ValueRange());
  ClassFieldsOp::create(builder, targetClass.getLoc(), rootObject.getResult(),
                        ArrayAttr());
  return wrapperClass;
}

struct ElaborateObjectPass
    : public circt::om::impl::ElaborateObjectBase<ElaborateObjectPass> {
  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbols(module);

    SmallVector<ClassOp> classes;
    for (auto classOp : module.getOps<ClassOp>()) {
      if (classOp.getSymName().starts_with("__om_elaborated_"))
        continue;
      if (!targetClass.empty() && classOp.getSymName() != targetClass)
        continue;
      if (classOp.getBodyBlock()->getNumArguments() == 0)
        classes.push_back(classOp);
    }

    if (!targetClass.empty() && classes.empty()) {
      module.emitError("om-elaborate-object could not find zero-input class ")
          << targetClass;
      return signalPassFailure();
    }

    for (auto classOp : classes) {
      auto wrapperClass = createTestWrapper(module, symbols, classOp);
      if (failed(wrapperClass) ||
          failed(applyElaborationTransform(*wrapperClass, symbols)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> circt::om::createElaborateObjectPass() {
  return std::make_unique<ElaborateObjectPass>();
}
