//===- OMLinkModules.cpp - OM Linker pass -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the OM Linker pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <memory>

namespace circt {
namespace om {
#define GEN_PASS_DEF_LINKMODULES
#include "circt/Dialect/OM/OMPasses.h.inc"
} // namespace om
} // namespace circt

using namespace circt;
using namespace om;
using namespace mlir;

namespace {
// A map from a pair of enclosing module op and old symbol to a new symbol.
using SymMappingTy =
    llvm::DenseMap<std::pair<ModuleOp, StringAttr>, StringAttr>;

struct ModuleInfo {
  ModuleInfo(mlir::ModuleOp module) : module(module) {
    block = std::make_unique<Block>();
  }

  // Populate `symbolToClasses`.
  LogicalResult initialize();

  // Update symbols based on the mapping and erase external and duplicate
  // classes.
  void postProcess(const SymMappingTy &symMapping,
                   const DenseSet<Operation *> &duplicateClasses);

  // A map from symbols to classes.
  llvm::DenseMap<StringAttr, ClassLike> symbolToClasses;

  // A target module.
  ModuleOp module;

  // A block to store original operations.
  std::unique_ptr<Block> block;
};

struct LinkModulesPass
    : public circt::om::impl::LinkModulesBase<LinkModulesPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

LogicalResult ModuleInfo::initialize() {
  for (auto &op : llvm::make_early_inc_range(module.getOps())) {
    if (auto classLike = dyn_cast<ClassLike>(op))
      symbolToClasses.insert({classLike.getSymNameAttr(), classLike});
    else {
      // Keep the op.
      op.moveBefore(block.get(), block->end());
    }
  }
  return success();
}

void ModuleInfo::postProcess(const SymMappingTy &symMapping,
                             const DenseSet<Operation *> &duplicateClasses) {
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      // Update class types when their symbols were renamed.
      [&](om::ClassType classType) -> std::pair<mlir::Type, WalkResult> {
        auto it = symMapping.find({module, classType.getClassName().getAttr()});
        // No change.
        if (it == symMapping.end())
          return {classType, WalkResult::skip()};
        return {om::ClassType::get(classType.getContext(),
                                   FlatSymbolRefAttr::get(it->second)),
                WalkResult::skip()};
      });

  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    // External classes and equivalent duplicate definitions must be erased.
    if (isa<ClassExternOp>(op) || duplicateClasses.contains(op)) {
      op->erase();
      return WalkResult::skip();
    }

    llvm::TypeSwitch<Operation *>(op)
        .Case<ClassOp>([&](ClassOp classOp) {
          auto it = symMapping.find({module, classOp.getNameAttr()});
          // Update its class name if changed.
          if (it != symMapping.end())
            classOp.setSymNameAttr(it->second);
          classOp.replaceFieldTypes(replacer);
        })
        .Case<ObjectOp, ElaboratedObjectOp>([&](auto objectLike) {
          auto it = symMapping.find(
              {module, objectLike.getClassNameAttr().getAttr()});
          // Update its class name if changed.
          if (it != symMapping.end())
            objectLike.setClassNameAttr(FlatSymbolRefAttr::get(it->second));
          replacer.replaceElementsIn(objectLike, false, false, true);
        });

    // Otherwise update om.class types.
    replacer.replaceElementsIn(op,
                               /*replaceAttrs=*/false,
                               /*replaceLocs=*/false,
                               /*replaceTypes=*/true);
    return WalkResult::advance();
  });
}

// Only coalesce definitions which forward constructor arguments to fields.
// Attributes (including parameter names and visibility), argument types, and
// the argument supplying each field must match exactly.
static bool haveEquivalentDefinitions(ClassOp lhs, ClassOp rhs) {
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
      !llvm::equal(lhs.getBodyBlock()->getArgumentTypes(),
                   rhs.getBodyBlock()->getArgumentTypes()) ||
      lhs.getFieldsOp()->getAttrDictionary() !=
          rhs.getFieldsOp()->getAttrDictionary())
    return false;
  return llvm::all_of_zip(lhs.getFieldsOp().getFields(),
                          rhs.getFieldsOp().getFields(), [](Value l, Value r) {
                            return cast<BlockArgument>(l).getArgNumber() ==
                                   cast<BlockArgument>(r).getArgNumber();
                          });
}

// Find names which will resolve to one definition: either a unique definition,
// or equivalent public field-only definitions. For the latter, referenced class
// types must also resolve consistently. Starting with all candidates and
// removing invalid ones handles recursive references.
static DenseSet<StringAttr>
findEquivalentClasses(const llvm::MapVector<StringAttr, SmallVector<ClassLike>>
                          &symbolToClasses) {
  DenseSet<StringAttr> equivalentClasses;
  llvm::MapVector<StringAttr, llvm::SetVector<StringAttr>> dependents;
  SmallVector<StringAttr> worklist;
  for (const auto &[name, classes] : symbolToClasses) {
    SmallVector<ClassOp> definitions;
    for (auto classLike : classes)
      if (auto classOp = dyn_cast<ClassOp>(classLike.getOperation()))
        definitions.push_back(classOp);
    // References to a unique definition already resolve to the same entity,
    // regardless of its visibility or the structure of its constructor.
    if (definitions.size() == 1) {
      equivalentClasses.insert(name);
      continue;
    }
    if (definitions.empty() || !llvm::all_of(definitions, [&](ClassOp classOp) {
          return classOp.isPublic() &&
                 llvm::hasSingleElement(
                     classOp.getBodyBlock()->getOperations()) &&
                 (classOp == definitions.front() ||
                  haveEquivalentDefinitions(definitions.front(), classOp));
        })) {
      worklist.push_back(name);
      continue;
    }

    bool resolvable = true;
    AttrTypeWalker walker;
    // Other symbol references are conservatively excluded. Non-OM symbols
    // remain in their original compilation units.
    walker.addWalk([&](hw::InnerRefAttr) { resolvable = false; });
    walker.addWalk([&](mlir::SymbolRefAttr) { resolvable = false; });
    walker.addWalk([&, name = name](ClassType type) {
      dependents[type.getClassName().getAttr()].insert(name);
      // Don't visit the class name as an arbitrary symbol reference.
      return WalkResult::skip();
    });
    // The definitions have identical attributes and types, so inspecting one
    // is sufficient. Field types are included in the class attributes.
    auto classOp = definitions.front();
    walker.walk<WalkOrder::PreOrder>(classOp->getAttrDictionary());
    walker.walk<WalkOrder::PreOrder>(
        classOp.getFieldsOp()->getAttrDictionary());
    for (auto type : classOp.getBodyBlock()->getArgumentTypes())
      walker.walk<WalkOrder::PreOrder>(type);
    if (resolvable)
      equivalentClasses.insert(name);
    else
      worklist.push_back(name);
  }
  for (const auto &[name, users] : dependents)
    if (!symbolToClasses.contains(name))
      worklist.push_back(name);
  while (!worklist.empty()) {
    auto name = worklist.pop_back_val();
    auto it = dependents.find(name);
    if (it == dependents.end())
      continue;
    for (auto user : it->second)
      if (equivalentClasses.erase(user))
        worklist.push_back(user);
  }
  return equivalentClasses;
}

// Return a failure if classes cannot be resolved. Return true if
// it's necessary to rename symbols.
static FailureOr<bool> resolveClasses(StringAttr name,
                                      ArrayRef<ClassLike> classes,
                                      bool hasEquivalentDefinitions) {
  bool existExternalClass = false;
  size_t countDefinition = 0;
  ClassOp classOp;

  for (auto op : classes) {
    if (isa<ClassExternOp>(op))
      existExternalClass = true;
    else {
      if (!classOp)
        classOp = cast<ClassOp>(op);
      ++countDefinition;
    }
  }

  // There must be exactly one definition if the symbol was referred by an
  // external class, unless the definitions are structurally equivalent.
  if (existExternalClass &&
      (countDefinition == 0 ||
       (countDefinition != 1 && !hasEquivalentDefinitions))) {
    SmallVector<Location> classExternLocs;
    SmallVector<Location> classLocs;
    for (auto op : classes)
      (isa<ClassExternOp>(op) ? classExternLocs : classLocs)
          .push_back(op.getLoc());

    auto diag = emitError(classExternLocs.front())
                << "class " << name << " is declared as an external class but "
                << (countDefinition == 0 ? "there is no definition"
                                         : "there are multiple definitions");
    for (auto loc : ArrayRef(classExternLocs).drop_front())
      diag.attachNote(loc) << "class " << name << " is declared here as well";

    if (countDefinition != 0) {
      // There are multiple definitions.
      for (auto loc : classLocs)
        diag.attachNote(loc) << "class " << name << " is defined here";
    }
    return failure();
  }

  if (!existExternalClass)
    return countDefinition != 1 && !hasEquivalentDefinitions;

  assert(classOp && (countDefinition == 1 || hasEquivalentDefinitions));

  // Raise errors if linked external modules are not compatible with the
  // definition.
  auto emitError = [&](Operation *op) {
    auto diag = op->emitError()
                << "failed to link class " << name
                << " since declaration doesn't match the definition: ";
    diag.attachNote(classOp.getLoc()) << "definition is here";
    return diag;
  };

  for (auto op : classes) {
    if (isa<ClassOp>(op))
      continue;

    if (classOp.getBodyBlock()->getNumArguments() !=
        op.getBodyBlock()->getNumArguments())
      return emitError(op) << "the number of arguments is not equal, "
                           << classOp.getBodyBlock()->getNumArguments()
                           << " vs " << op.getBodyBlock()->getNumArguments();
    unsigned index = 0;
    for (auto [l, r] : llvm::zip(classOp.getBodyBlock()->getArgumentTypes(),
                                 op.getBodyBlock()->getArgumentTypes())) {
      if (l != r)
        return emitError(op) << index << "-th argument type is not equal, " << l
                             << " vs " << r;
      index++;
    }
    // Check declared fields.
    llvm::DenseSet<StringAttr> declaredFields;

    for (auto nameAttr : op.getFieldNames()) {
      StringAttr name = cast<StringAttr>(nameAttr);
      std::optional<Type> opTypeOpt = op.getFieldType(name);

      if (!opTypeOpt.has_value())
        return emitError(op) << " no type for field " << name;
      Type opType = opTypeOpt.value();

      std::optional<Type> classTypeOpt = classOp.getFieldType(name);

      // Field not found in its definition.
      if (!classTypeOpt.has_value())
        return emitError(op) << "declaration has a field " << name
                             << " but not found in its definition";
      Type classType = classTypeOpt.value();

      if (classType != opType)
        return emitError(op)
               << "declaration has a field " << name
               << " but types don't match, " << classType << " vs " << opType;
      declaredFields.insert(name);
    }

    for (auto fieldName : classOp.getFieldNames())
      if (!declaredFields.count(cast<StringAttr>(fieldName)))
        return emitError(op) << "definition has a field " << fieldName
                             << " but not found in this declaration";
  }
  return false;
}

void LinkModulesPass::runOnOperation() {
  auto toplevelModule = getOperation();
  // 1. Initialize ModuleInfo.
  SmallVector<ModuleInfo> modules;
  size_t counter = 0;
  for (auto module : toplevelModule.getOps<ModuleOp>()) {
    auto name = module->getAttrOfType<StringAttr>("om.namespace");
    // Use `counter` if the namespace is not specified beforehand.
    if (!name) {
      name = StringAttr::get(module.getContext(), "module_" + Twine(counter++));
      module->setAttr("om.namespace", name);
    }
    modules.emplace_back(module);
  }

  if (failed(failableParallelForEach(&getContext(), modules, [](auto &info) {
        // Collect local information.
        return info.initialize();
      })))
    return signalPassFailure();

  // 2. Symbol resolution. Coalesce equivalent field-only definitions, resolve
  //    external symbols, and rename other colliding symbols if necessary.

  // Global namespace to get unique names to symbols.
  Namespace nameSpace;
  // A map from a pair of enclosing module op and old symbol to a new symbol.
  SymMappingTy symMapping;

  // Construct a global map from symbols to class operations.
  llvm::MapVector<StringAttr, SmallVector<ClassLike>> symbolToClasses;
  for (const auto &info : modules)
    for (auto &[name, op] : info.symbolToClasses) {
      symbolToClasses[name].push_back(op);
      // Add names to avoid collision.
      (void)nameSpace.newName(name.getValue());
    }

  // Equivalent public field-only definitions can be shared across compilation
  // units. Other symbols used as external classes must have exactly one
  // definition.
  auto equivalentClasses = findEquivalentClasses(symbolToClasses);
  DenseSet<Operation *> duplicateClasses;
  for (auto &[name, classes] : symbolToClasses) {
    // Check if it's legal to link classes. `resolveClasses` returns true if
    // it's necessary to rename symbols.
    auto result =
        resolveClasses(name, classes, equivalentClasses.contains(name));
    if (failed(result))
      return signalPassFailure();

    // Keep just the first of the equivalent definitions. All references retain
    // the original name, so both constructor calls and class types resolve to
    // the same definition after flattening.
    if (equivalentClasses.contains(name)) {
      bool foundDefinition = false;
      for (auto op : classes)
        if (isa<ClassOp>(op)) {
          if (foundDefinition)
            duplicateClasses.insert(op.getOperation());
          foundDefinition = true;
        }
    }

    // We can resolve symbol collision for symbols not referred by external
    // classes. Create a new name using `om.namespace` attributes as a
    // suffix.
    if (*result)
      for (auto op : classes) {
        auto enclosingModule = cast<mlir::ModuleOp>(op->getParentOp());
        auto nameSpaceId =
            enclosingModule->getAttrOfType<StringAttr>("om.namespace");
        symMapping[{enclosingModule, name}] = StringAttr::get(
            &getContext(),
            nameSpace.newName(name.getValue(), nameSpaceId.getValue()));
      }
  }

  // 3. Post-processing. Update class names and erase external and duplicate
  //    classes.
  parallelForEach(&getContext(), modules, [&](auto &info) {
    info.postProcess(symMapping, duplicateClasses);
  });

  // Finally move operations to the toplevel module.
  auto *block = toplevelModule.getBody();
  for (auto &info : modules) {
    block->getOperations().splice(block->end(),
                                  info.module.getBody()->getOperations());
    // Restore non-OM operations.
    assert(info.module.getBody()->empty());
    info.module.getBody()->getOperations().splice(
        info.module.getBody()->begin(), info.block->getOperations());
  }
}
