//===- CheckDomain.cpp - Verify clock domains ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "seq-check-domain"

namespace circt::seq {
#define GEN_PASS_DEF_CHECKDOMAIN
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace circt::seq

using namespace circt;
using namespace circt::seq;

namespace {

static const StringRef innerSymAttrName =
    hw::InnerSymbolTable::getInnerSymbolAttrName();

/// Inliner used to flatten only the cloned module being checked. Unlike the
/// general HW flattening pass, this does not consult or modify hw.hierpath ops:
/// those paths continue to refer to the original hierarchy.
struct CheckDomainInliner : public mlir::InlinerInterface {
  StringRef prefix;
  DenseMap<StringAttr, StringAttr> *symMapping;
  mlir::AttrTypeReplacer *replacer;

  CheckDomainInliner(MLIRContext *context, StringRef prefix,
                     DenseMap<StringAttr, StringAttr> *symMapping,
                     mlir::AttrTypeReplacer *replacer)
      : InlinerInterface(context), prefix(prefix), symMapping(symMapping),
        replacer(replacer) {}

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  void handleTerminator(Operation *op,
                        mlir::ValueRange valuesToRepl) const override {
    assert(isa<hw::OutputOp>(op));
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }

  bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const final {
    return true;
  }

  StringAttr updateName(StringAttr attr) const {
    if (attr.getValue().empty())
      return attr;
    return StringAttr::get(attr.getContext(), prefix + "/" + attr.getValue());
  }

  void processInlinedBlocks(
      iterator_range<Region::iterator> inlinedBlocks) override {
    for (Block &block : inlinedBlocks)
      block.walk([&](Operation *op) {
        if (auto name = op->getAttrOfType<StringAttr>("name"))
          op->setAttr("name", updateName(name));
        if (auto name = op->getAttrOfType<StringAttr>("instanceName"))
          op->setAttr("instanceName", updateName(name));
        if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names")) {
          SmallVector<Attribute> names(namesAttr.getValue().begin(),
                                       namesAttr.getValue().end());
          for (auto &name : names)
            if (auto nameStr = dyn_cast<StringAttr>(name))
              name = updateName(nameStr);
          op->setAttr("names", ArrayAttr::get(namesAttr.getContext(), names));
        }

        if (auto innerSymAttr =
                op->getAttrOfType<hw::InnerSymAttr>(innerSymAttrName)) {
          auto it = symMapping->find(innerSymAttr.getSymName());
          if (it != symMapping->end())
            op->setAttr(innerSymAttrName, hw::InnerSymAttr::get(it->second));
        }
        replacer->replaceElementsIn(op);
      });
  }
};

LogicalResult inlineCheckDomainInstances(hw::HWModuleOp module,
                                         mlir::ModuleOp top) {
  mlir::InlinerConfig config;
  mlir::SymbolTable symbolTable(top);
  hw::InnerSymbolNamespace ns(module);

  while (true) {
    SmallVector<hw::InstanceOp> instances;
    module.walk([&](hw::InstanceOp instance) {
      if (symbolTable.lookup<hw::HWModuleOp>(
              instance.getModuleNameAttr().getValue()))
        instances.push_back(instance);
    });
    if (instances.empty())
      return success();

    for (auto instance : instances) {
      auto sourceModule = symbolTable.lookup<hw::HWModuleOp>(
          instance.getModuleNameAttr().getValue());
      if (!sourceModule)
        continue;
      if (sourceModule == module) {
        instance.emitError("cannot flatten a recursive HW module instance");
        return failure();
      }

      DenseMap<StringAttr, StringAttr> oldToNewInnerSyms;
      sourceModule.walk([&](Operation *op) {
        if (auto innerSymAttr =
                op->getAttrOfType<hw::InnerSymAttr>(innerSymAttrName))
          oldToNewInnerSyms.try_emplace(
              innerSymAttr.getSymName(),
              StringAttr::get(
                  module.getContext(),
                  ns.newName(innerSymAttr.getSymName().getValue())));
      });

      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](hw::InnerRefAttr attr) -> std::pair<Attribute, WalkResult> {
            if (attr.getModule() != sourceModule.getModuleNameAttr())
              return {attr, WalkResult::skip()};

            auto it = oldToNewInnerSyms.find(attr.getName());
            if (it == oldToNewInnerSyms.end())
              return {attr, WalkResult::skip()};

            return {
                hw::InnerRefAttr::get(module.getModuleNameAttr(), it->second),
                WalkResult::skip()};
          });

      CheckDomainInliner inliner(module.getContext(),
                                 instance.getInstanceName(), &oldToNewInnerSyms,
                                 &replacer);
      if (failed(mlir::inlineRegion(
              inliner, config.getCloneCallback(), &sourceModule.getBody(),
              instance, instance.getOperands(), instance.getResults(),
              std::nullopt, /*shouldClone=*/true))) {
        instance.emitError("failed to inline '")
            << sourceModule.getModuleName() << "' into instance '"
            << instance.getInstanceName() << "'";
        return failure();
      }
      instance.erase();
    }
  }
}

/// Drop wire symbols from the temporary flattened module. Wires only preserve
/// an SSA edge, so keeping their inner symbols prevents canonicalization from
/// folding them away. The original module and its hierarchical paths are not
/// visited by this helper.
static void eraseClonedWireSymbols(hw::HWModuleOp module) {
  module.walk([](hw::WireOp wire) { wire->removeAttr(innerSymAttrName); });
}

struct CheckDomainPass : public impl::CheckDomainBase<CheckDomainPass> {
  using Base::Base;

  void runOnOperation() override;

private:
  LogicalResult checkValue(CheckClockDomainOp check, Value value, Value clock,
                           llvm::SmallPtrSetImpl<Value> &visited);
  static std::string describeValue(Value value);
};

std::string CheckDomainPass::describeValue(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto module =
            dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp()))
      return "module input '" +
             module.getArgName(blockArg.getArgNumber()).getValue().str() + "'";
    return ("block argument #" + Twine(blockArg.getArgNumber())).str();
  }

  if (auto *definingOp = value.getDefiningOp()) {
    if (auto name = definingOp->getAttrOfType<StringAttr>("name"))
      return ("value '" + name.getValue() + "' (defined by '" +
              definingOp->getName().getStringRef() + "')")
          .str();
    if (auto namehint = definingOp->getAttrOfType<StringAttr>("sv.namehint"))
      return ("value '" + namehint.getValue() + "' (defined by '" +
              definingOp->getName().getStringRef() + "')")
          .str();
    return ("result of '" + definingOp->getName().getStringRef() + "'").str();
  }
  return "value";
}

LogicalResult
CheckDomainPass::checkValue(CheckClockDomainOp check, Value value, Value clock,
                            llvm::SmallPtrSetImpl<Value> &visited) {
  auto checkClocked = [&](Operation *op) -> LogicalResult {
    auto clocked = dyn_cast<Clocked>(op);
    if (!clocked || clocked.getClk() == clock)
      return success();

    check.emitOpError() << "input depends on a sequential element clocked by "
                        << describeValue(clocked.getClk())
                        << ", not the expected clock " << describeValue(clock);
    return failure();
  };

  struct WorkItem {
    Value value;
    bool walkOperands;
    bool walkUsers;
  };

  SmallVector<WorkItem> worklist{
      {value, /*walkOperands=*/true, /*walkUsers=*/true}};
  while (!worklist.empty()) {
    auto [current, walkOperands, walkUsers] = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    if (auto *definingOp = current.getDefiningOp()) {
      if (failed(checkClocked(definingOp)))
        return failure();
      if (definingOp->hasTrait<mlir::OpTrait::ConstantLike>())
        continue;

      // A clocked operation is a domain boundary. In particular, do not walk
      // through its data, clock, or reset operands: those operands can belong
      // to a different domain from the value it produces.
      if (walkOperands && !isa<Clocked>(definingOp))
        for (Value operand : definingOp->getOperands())
          worklist.push_back(
              {operand, /*walkOperands=*/true, /*walkUsers=*/false});
    }

    if (walkUsers) {
      for (Operation *user : current.getUsers()) {
        if (failed(checkClocked(user)))
          return failure();
        for (Value result : user->getResults())
          worklist.push_back(
              {result, /*walkOperands=*/false, /*walkUsers=*/true});
      }
    }
  }
  return success();
}

void CheckDomainPass::runOnOperation() {
  SmallVector<sv::BindOp> binds;
  getOperation()->walk([&](sv::BindOp bind) { binds.push_back(bind); });
  for (auto bind : binds)
    bind.erase();

  if (moduleName.empty()) {
    emitError(getOperation().getLoc()) << "requires --module-name";
    return signalPassFailure();
  }

  mlir::SymbolTable symbolTable(getOperation());
  auto module = symbolTable.lookup<hw::HWModuleOp>(moduleName);
  if (!module) {
    emitError(getOperation().getLoc())
        << "could not find HW module '" << moduleName << "'";
    return signalPassFailure();
  }

  auto flattenedName = moduleName + "_flatten";
  if (symbolTable.lookup(flattenedName)) {
    emitError(getOperation().getLoc())
        << "cannot create flattened module '" << flattenedName
        << "': symbol already exists";
    return signalPassFailure();
  }

  // Work on a clone so that flattening does not modify the hierarchy referred
  // to by existing hierarchical paths.
  auto flattenedModule = module.clone();
  flattenedModule.setName(StringAttr::get(&getContext(), flattenedName));

  // A cloned operation can still contain inner references to the original
  // module. Retarget those references to the clone before inserting it.
  auto originalModuleName = module.getModuleNameAttr();
  auto flattenedModuleName = flattenedModule.getModuleNameAttr();
  mlir::AttrTypeReplacer cloneReplacer;
  cloneReplacer.addReplacement(
      [&](hw::InnerRefAttr attr) -> std::pair<Attribute, WalkResult> {
        if (attr.getModule() != originalModuleName)
          return {attr, WalkResult::skip()};
        return {hw::InnerRefAttr::get(flattenedModuleName, attr.getName()),
                WalkResult::skip()};
      });
  flattenedModule.walk(
      [&](Operation *op) { cloneReplacer.replaceElementsIn(op); });

  getOperation().push_back(flattenedModule);
  module = flattenedModule;

  // Flatten only the clone. This deliberately does not use FlattenModules:
  // that pass operates on the complete instance graph and would also modify
  // modules and hierarchical paths belonging to the original design.
  if (failed(inlineCheckDomainInstances(module, getOperation())))
    return signalPassFailure();

  eraseClonedWireSymbols(module);

  mlir::OpPassManager pipeline("hw.module");
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
  if (failed(runPipeline(pipeline, module)))
    return signalPassFailure();
  LogicalResult result = success();
  module.walk([&](CheckClockDomainOp check) {
    llvm::SmallPtrSet<Value, 32> visited;
    if (failed(checkValue(check, check.getInput(), check.getClock(), visited)))
      result = failure();
  });
  if (failed(result))
    signalPassFailure();
}

} // namespace
