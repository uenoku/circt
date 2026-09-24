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
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

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
  unsigned numInlined = 0;

  while (true) {
    SmallVector<hw::InstanceOp> instances;
    module.walk([&](hw::InstanceOp instance) {
      if (symbolTable.lookup<hw::HWModuleOp>(
              instance.getModuleNameAttr().getValue()))
        instances.push_back(instance);
    });
    if (instances.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "inlined " << numInlined << " HW instance(s)\n");
      return success();
    }

    for (auto instance : instances) {
      auto sourceModule = symbolTable.lookup<hw::HWModuleOp>(
          instance.getModuleNameAttr().getValue());
      if (!sourceModule)
        continue;
      if (sourceModule == module) {
        instance.emitError("cannot flatten a recursive HW module instance");
        return failure();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "inlining instance '" << instance.getInstanceName()
                 << "' of '" << sourceModule.getModuleName() << "'\n");

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
      ++numInlined;
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

/// Replace EICG wrapper instances in the cloned module with seq.clock_gate.
/// This mirrors the legalization done by the Arc StripSV pass, but is scoped
/// to the temporary clone so that the original design and its hierpaths are
/// left unchanged.
static LogicalResult legalizeClockGates(hw::HWModuleOp module,
                                        mlir::ModuleOp top) {
  mlir::SymbolTable symbolTable(top);
  SmallVector<hw::InstanceOp> instances;
  module.walk([&](hw::InstanceOp instance) { instances.push_back(instance); });

  auto *context = module.getContext();
  auto expectedInputNames =
      ArrayAttr::get(context, {StringAttr::get(context, "in"),
                               StringAttr::get(context, "test_en"),
                               StringAttr::get(context, "en")});
  auto expectedOutputNames =
      ArrayAttr::get(context, {StringAttr::get(context, "out")});
  auto i1Type = IntegerType::get(context, 1);

  SmallPtrSet<Operation *, 4> checkedModules;
  unsigned numLegalized = 0;
  for (auto instance : instances) {
    auto external = symbolTable.lookup<hw::HWModuleExternOp>(
        instance.getModuleNameAttr().getValue());
    if (!external || external.getVerilogModuleName() != "EICG_wrapper")
      continue;

    if (checkedModules.insert(external.getOperation()).second) {
      if (!llvm::equal(external.getInputNames(), expectedInputNames) ||
          !llvm::equal(external.getOutputNames(), expectedOutputNames)) {
        external.emitError("clock gate module `")
            << external.getModuleName() << "` has incompatible port names "
            << external.getInputNames() << " -> " << external.getOutputNames();
        return failure();
      }

      auto inputTypes = external.getInputTypes();
      auto outputTypes = external.getOutputTypes();
      bool clockPorts = inputTypes.size() == 3 && outputTypes.size() == 1 &&
                        isa<ClockType>(inputTypes[0]) &&
                        isa<ClockType>(outputTypes[0]);
      bool bitPorts = inputTypes.size() == 3 && outputTypes.size() == 1 &&
                      inputTypes[0] == i1Type && outputTypes[0] == i1Type;
      if (!clockPorts && !bitPorts) {
        external.emitError("clock gate module `")
            << external.getModuleName() << "` has incompatible port types "
            << external.getInputTypes() << " -> " << external.getOutputTypes();
        return failure();
      }
      if (inputTypes[1] != i1Type || inputTypes[2] != i1Type) {
        external.emitError("clock gate module `")
            << external.getModuleName()
            << "` has incompatible enable port types "
            << external.getInputTypes() << " -> " << external.getOutputTypes();
        return failure();
      }
    }

    if (instance.getNumOperands() != 3 || instance.getNumResults() != 1) {
      instance.emitError("expected EICG_wrapper instance to have three inputs "
                         "and one output");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "legalizing EICG instance '" << instance.getInstanceName()
               << "' in '" << module.getModuleName() << "'\n");

    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    Value input = instance.getOperand(0);
    if (isa<IntegerType>(input.getType()))
      input = ToClockOp::create(builder, input);

    auto gated =
        ClockGateOp::create(builder, input, instance.getOperand(1),
                            instance.getOperand(2), hw::InnerSymAttr{});
    Value output = gated;
    if (isa<IntegerType>(instance.getResult(0).getType()))
      output = FromClockOp::create(builder, gated);

    instance.getResult(0).replaceAllUsesWith(output);
    instance.erase();
    ++numLegalized;
  }
  LLVM_DEBUG(llvm::dbgs() << "legalized " << numLegalized
                          << " EICG instance(s)\n");
  return success();
}

struct CheckDomainPass : public impl::CheckDomainBase<CheckDomainPass> {
  using Base::Base;

  void runOnOperation() override;

private:
  LogicalResult checkValue(Operation *check, Value value, Value clock,
                           llvm::SmallPtrSetImpl<Value> &visited,
                           bool allowCrossing, bool skipClockGate,
                           bool requireSameClock);
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
    if (auto instanceName =
            definingOp->getAttrOfType<StringAttr>("instanceName"))
      return ("value '" + instanceName.getValue() + "' (defined by '" +
              definingOp->getName().getStringRef() + "')")
          .str();
    if (!isa<hw::InstanceOp>(definingOp))
      if (auto namehint = definingOp->getAttrOfType<StringAttr>("sv.namehint"))
        return ("value '" + namehint.getValue() + "' (defined by '" +
                definingOp->getName().getStringRef() + "')")
            .str();
    return ("result of '" + definingOp->getName().getStringRef() + "'").str();
  }
  return "value";
}

LogicalResult CheckDomainPass::checkValue(Operation *check, Value value,
                                          Value clock,
                                          llvm::SmallPtrSetImpl<Value> &visited,
                                          bool allowCrossing,
                                          bool skipClockGate,
                                          bool requireSameClock) {
  auto normalizeClock = [&](Value clock) {
    while (true) {
      if (auto clockGate = clock.getDefiningOp<ClockGateOp>()) {
        clock = clockGate.getInput();
        continue;
      }
      if (auto toClock = clock.getDefiningOp<ToClockOp>()) {
        clock = toClock.getInput();
        continue;
      }
      if (auto fromClock = clock.getDefiningOp<FromClockOp>()) {
        clock = fromClock.getInput();
        continue;
      }
      return clock;
    }
  };

  auto isClockGateResult = [&](Value clock) {
    while (true) {
      if (auto toClock = clock.getDefiningOp<ToClockOp>()) {
        clock = toClock.getInput();
        continue;
      }
      if (auto fromClock = clock.getDefiningOp<FromClockOp>()) {
        clock = fromClock.getInput();
        continue;
      }
      return clock.getDefiningOp<ClockGateOp>() != nullptr;
    }
  };

  auto checkClocked = [&](Operation *op) -> LogicalResult {
    auto clocked = dyn_cast<Clocked>(op);
    if (!clocked)
      return success();

    if (skipClockGate && isClockGateResult(clocked.getClk()))
      return success();

    bool clocksMatch = clocked.getClk() == clock ||
                       (skipClockGate && normalizeClock(clocked.getClk()) ==
                                             normalizeClock(clock));
    if (requireSameClock) {
      if (allowCrossing || clocksMatch)
        return success();

      check->emitOpError()
          << "input depends on a sequential element clocked by "
          << describeValue(clocked.getClk()) << ", not the expected clock "
          << describeValue(clock);
      return failure();
    }

    if (!clocksMatch)
      return success();

    check->emitOpError() << "input depends on a sequential element clocked by "
                         << describeValue(clocked.getClk())
                         << ", but it must not depend on the expected clock "
                         << describeValue(clock);
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
        if (skipClockGate) {
          if (isa<ClockGateOp>(user))
            continue;
          if (auto clocked = dyn_cast<Clocked>(user))
            if (isClockGateResult(clocked.getClk()))
              continue;
        }
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
  LLVM_DEBUG(llvm::dbgs() << "starting clock-domain check for module '"
                          << moduleName << "'\n");
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

  LLVM_DEBUG(llvm::dbgs() << "created temporary module '"
                          << flattenedModule.getModuleName() << "'\n");

  // Flatten only the clone. This deliberately does not use FlattenModules:
  // that pass operates on the complete instance graph and would also modify
  // modules and hierarchical paths belonging to the original design.
  if (failed(inlineCheckDomainInstances(module, getOperation())))
    return signalPassFailure();

  LLVM_DEBUG(llvm::dbgs() << "finished inlining in '" << module.getModuleName()
                          << "'\n");

  if (failed(legalizeClockGates(module, getOperation())))
    return signalPassFailure();

  eraseClonedWireSymbols(module);

  mlir::OpPassManager pipeline("hw.module");
  pipeline.addPass(mlir::createCanonicalizerPass());
  pipeline.addPass(mlir::createCSEPass());
  if (failed(runPipeline(pipeline, module)))
    return signalPassFailure();
  LLVM_DEBUG(llvm::dbgs() << "finished canonicalization in '"
                          << module.getModuleName() << "'\n");
  LogicalResult result = success();
  unsigned numChecks = 0;
  module.walk([&](CheckClockDomainOp check) {
    ++numChecks;
    LLVM_DEBUG(llvm::dbgs()
               << "checking seq.check_clock_domain #" << numChecks << "\n");
    llvm::SmallPtrSet<Value, 32> visited;
    if (failed(checkValue(check.getOperation(), check.getInput(),
                          check.getClock(), visited, allowCrossing,
                          skipClockGate, /*requireSameClock=*/true)))
      result = failure();
  });
  module.walk([&](CheckClockDomainNeqOp check) {
    ++numChecks;
    LLVM_DEBUG(llvm::dbgs()
               << "checking seq.check_clock_domain_neq #" << numChecks << "\n");
    llvm::SmallPtrSet<Value, 32> visited;
    if (failed(checkValue(check.getOperation(), check.getInput(),
                          check.getClock(), visited, allowCrossing,
                          skipClockGate, /*requireSameClock=*/false)))
      result = failure();
  });
  LLVM_DEBUG(llvm::dbgs() << "finished " << numChecks
                          << " clock-domain check(s)\n");
  if (failed(result))
    signalPassFailure();
}

} // namespace
