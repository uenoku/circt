//===- InnerSymbolTable.cpp - InnerSymbolTable and InnerRef verification --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements InnerSymbolTable and verification for InnerRef's.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace hw;

namespace circt {
namespace hw {

//===----------------------------------------------------------------------===//
// InnerSymbolTable
//===----------------------------------------------------------------------===//
InnerSymbolTable::InnerSymbolTable(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>());
  // Save the operation this table is for.
  this->innerSymTblOp = op;

  walkSymbols(op, [&](StringAttr name, const InnerSymTarget &target) {
    auto it = symbolTable.try_emplace(name, target);
    (void)it;
    assert(it.second && "repeated symbol found");
  });
}

FailureOr<InnerSymbolTable> InnerSymbolTable::get(Operation *op) {
  return get(op, nullptr);
}

FailureOr<InnerSymbolTable>
InnerSymbolTable::get(Operation *op,
                      SmallVectorImpl<Operation *> *innerRefUsers) {
  assert(op);
  if (!op->hasTrait<OpTrait::InnerSymbolTable>())
    return op->emitError("expected operation to have InnerSymbolTable trait");

  TableTy table;
  auto result = walkSymbols(
      op,
      [&](StringAttr name, const InnerSymTarget &target) -> LogicalResult {
        auto it = table.try_emplace(name, target);
        if (it.second)
          return success();
        auto existing = it.first->second;
        return target.getOp()
            ->emitError()
            .append("redefinition of inner symbol named '", name.strref(), "'")
            .attachNote(existing.getOp()->getLoc())
            .append("see existing inner symbol definition here");
      },
      innerRefUsers);
  if (failed(result))
    return failure();
  return InnerSymbolTable(op, std::move(table));
}

LogicalResult InnerSymbolTable::walkSymbols(Operation *op,
                                            InnerSymCallbackFn callback) {
  return walkSymbols(op, callback, nullptr);
}

LogicalResult
InnerSymbolTable::walkSymbols(Operation *op, InnerSymCallbackFn callback,
                              SmallVectorImpl<Operation *> *innerRefUsers) {
  auto walkSym = [&](StringAttr name, const InnerSymTarget &target) {
    assert(name && !name.getValue().empty());
    return callback(name, target);
  };

  auto walkSyms = [&](hw::InnerSymAttr symAttr,
                      const InnerSymTarget &baseTarget) -> LogicalResult {
    assert(baseTarget.getField() == 0);
    for (auto symProp : symAttr) {
      if (failed(walkSym(symProp.getName(),
                         InnerSymTarget::getTargetForSubfield(
                             baseTarget, symProp.getFieldID()))))
        return failure();
    }
    return success();
  };

  // Check for ports
  if (auto mod = dyn_cast<PortList>(op)) {
    for (auto [i, port] : llvm::enumerate(mod.getPortList())) {
      if (auto symAttr = port.getSym())
        if (failed(walkSyms(symAttr, InnerSymTarget(i, mod))))
          return failure();
    }
  }

  // Walk the operation and add InnerSymbolTarget's to the table.
  auto walkSymbolOp = [&](Operation *curOp) -> WalkResult {
    if (auto symOp = dyn_cast<InnerSymbolOpInterface>(curOp))
      if (auto symAttr = symOp.getInnerSymAttr())
        if (failed(walkSyms(symAttr, InnerSymTarget(symOp))))
          return WalkResult::interrupt();
    return WalkResult::advance();
  };
  if (!innerRefUsers)
    return success(
        !op->walk<mlir::WalkOrder::PreOrder>(walkSymbolOp).wasInterrupted());

  // Collect users while building the table, rather than walking the body again
  // once all tables have been built. Keep the original visitation orders for
  // both duplicate-symbol diagnostics and inner-reference verification.
  return success(!op->walk([&](Operation *curOp,
                               const mlir::WalkStage &stage) -> WalkResult {
                      if (stage.isBeforeAllRegions() &&
                          walkSymbolOp(curOp).wasInterrupted())
                        return WalkResult::interrupt();
                      if (stage.isAfterAllRegions() &&
                          isa<InnerRefUserOpInterface>(curOp))
                        innerRefUsers->push_back(curOp);
                      return WalkResult::advance();
                    }).wasInterrupted());
}

/// Look up a symbol with the specified name, returning empty InnerSymTarget if
/// no such name exists. Names never include the @ on them.
InnerSymTarget InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}
InnerSymTarget InnerSymbolTable::lookup(StringAttr name) const {
  return symbolTable.lookup(name);
}

/// Look up a symbol with the specified name, returning null if no such
/// name exists or doesn't target just an operation.
Operation *InnerSymbolTable::lookupOp(StringRef name) const {
  return lookupOp(StringAttr::get(innerSymTblOp->getContext(), name));
}
Operation *InnerSymbolTable::lookupOp(StringAttr name) const {
  auto result = lookup(name);
  if (result.isOpOnly())
    return result.getOp();
  return nullptr;
}

/// Get InnerSymbol for an operation.
StringAttr InnerSymbolTable::getInnerSymbol(Operation *op) {
  if (auto innerSymOp = dyn_cast<InnerSymbolOpInterface>(op))
    return innerSymOp.getInnerNameAttr();
  return {};
}

/// Get InnerSymbol for a target.  Be robust to queries on unexpected
/// operations to avoid users needing to know the details.
StringAttr InnerSymbolTable::getInnerSymbol(const InnerSymTarget &target) {
  // Assert on misuse, but try to handle queries otherwise.
  assert(target);

  // Obtain the base InnerSymAttr for the specified target.
  auto getBase = [](auto &target) -> hw::InnerSymAttr {
    if (target.isPort()) {
      if (auto mod = dyn_cast<PortList>(target.getOp())) {
        assert(target.getPort() < mod.getNumPorts());
        return mod.getPort(target.getPort()).getSym();
      }
    } else {
      // InnerSymbols only supported if op implements the interface.
      if (auto symOp = dyn_cast<InnerSymbolOpInterface>(target.getOp()))
        return symOp.getInnerSymAttr();
    }
    return {};
  };

  if (auto base = getBase(target))
    return base.getSymIfExists(target.getField());
  return {};
}

//===----------------------------------------------------------------------===//
// InnerSymbolTableCollection
//===----------------------------------------------------------------------===//

InnerSymbolTable &
InnerSymbolTableCollection::getInnerSymbolTable(Operation *op) {
  auto it = symbolTables.try_emplace(op, nullptr);
  if (it.second)
    it.first->second = ::std::make_unique<InnerSymbolTable>(op);
  return *it.first->second;
}

LogicalResult
InnerSymbolTableCollection::populateAndVerifyTables(Operation *innerRefNSOp) {
  return populateAndVerifyTables(innerRefNSOp, {});
}

LogicalResult InnerSymbolTableCollection::populateAndVerifyTables(
    Operation *innerRefNSOp,
    llvm::function_ref<void(Operation *, ArrayRef<Operation *>)>
        collectInnerRefUsers) {
  // Gather top-level operations that have the InnerSymbolTable trait.
  SmallVector<Operation *> innerSymTableOps(llvm::make_filter_range(
      llvm::make_pointer_range(innerRefNSOp->getRegion(0).front()),
      [&](Operation *op) {
        return op->hasTrait<OpTrait::InnerSymbolTable>();
      }));

  // Ensure entries exist for each operation.
  llvm::for_each(innerSymTableOps,
                 [&](auto *op) { symbolTables.try_emplace(op, nullptr); });

  // Construct the tables in parallel (if context allows it).
  return mlir::failableParallelForEach(
      innerRefNSOp->getContext(), innerSymTableOps, [&](auto *op) {
        auto it = symbolTables.find(op);
        assert(it != symbolTables.end());
        if (!it->second) {
          SmallVector<Operation *> innerRefUsers;
          auto result = InnerSymbolTable::get(
              op, collectInnerRefUsers ? &innerRefUsers : nullptr);
          if (failed(result))
            return failure();
          it->second = std::make_unique<InnerSymbolTable>(std::move(*result));
          if (collectInnerRefUsers)
            collectInnerRefUsers(op, innerRefUsers);
          return success();
        }
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace
//===----------------------------------------------------------------------===//

InnerSymTarget InnerRefNamespace::lookup(hw::InnerRefAttr inner) const {
  auto *mod = symTable.lookup(inner.getModule());
  if (!mod)
    return {};
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookup(inner.getName());
}

Operation *InnerRefNamespace::lookupOp(hw::InnerRefAttr inner) const {
  auto *mod = symTable.lookup(inner.getModule());
  if (!mod)
    return nullptr;
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookupOp(inner.getName());
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace verification
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyInnerRefNamespace(Operation *op) {
  // Pre-populate entries so parallel table construction can collect users
  // without inserting into a shared map.
  DenseMap<Operation *, SmallVector<Operation *>> innerRefUsers;
  SmallVector<Operation *> topLevelOps;
  SmallVector<Operation *> leafOps;
  for (auto &child : op->getRegion(0).front()) {
    if (child.hasTrait<OpTrait::InnerSymbolTable>())
      innerRefUsers.try_emplace(&child);
    if (child.getNumRegions() != 0)
      topLevelOps.push_back(&child);
    else
      leafOps.push_back(&child);
  }

  // Construct the symbol tables and collect their users in the same walk. All
  // tables must be complete before verifying users, which may refer to symbols
  // defined later in the body or in another module.
  InnerSymbolTableCollection innerSymTables;
  if (failed(innerSymTables.populateAndVerifyTables(
          op, [&](Operation *tableOp, ArrayRef<Operation *> users) {
            innerRefUsers.find(tableOp)->second.append(users.begin(),
                                                       users.end());
          })))
    return failure();

  SymbolTable symbolTable(op);
  InnerRefNamespace ns{symbolTable, innerSymTables};

  // Verify collected users in parallel. Top-level children without an inner
  // symbol table still need a walk to discover their users.
  auto verifySymbolUserFn = [&](Operation *op) -> WalkResult {
    if (auto user = dyn_cast<InnerRefUserOpInterface>(op))
      return WalkResult(user.verifyInnerRefs(ns));
    return WalkResult::advance();
  };

  // Handle regionless operations directly -- not worth parallelizing.
  for (auto *leaf : leafOps)
    if (verifySymbolUserFn(leaf).wasInterrupted())
      return failure();
  return mlir::failableParallelForEach(
      op->getContext(), topLevelOps, [&](Operation *op) {
        auto it = innerRefUsers.find(op);
        if (it == innerRefUsers.end())
          return success(!op->walk(verifySymbolUserFn).wasInterrupted());
        for (auto *user : it->second)
          if (failed(cast<InnerRefUserOpInterface>(user).verifyInnerRefs(ns)))
            return failure();
        return success();
      });
}

} // namespace detail

bool InnerRefNamespaceLike::classof(mlir::Operation *op) {
  return op->hasTrait<mlir::OpTrait::InnerRefNamespace>() ||
         op->hasTrait<mlir::OpTrait::SymbolTable>();
}

bool InnerRefNamespaceLike::classof(
    const mlir::RegisteredOperationName *opInfo) {
  return opInfo->hasTrait<mlir::OpTrait::InnerRefNamespace>() ||
         opInfo->hasTrait<mlir::OpTrait::SymbolTable>();
}

} // namespace hw
} // namespace circt
