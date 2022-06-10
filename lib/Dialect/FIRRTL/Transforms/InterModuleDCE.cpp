//===- InterModuleDCE.cpp - Remove Dead Ports ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-unused-ports"

using namespace circt;
using namespace firrtl;

/// Return true if this is a wire or a register or a node.
static bool isWireOrRegOrNode(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp, NodeOp>(op);
}

namespace {
struct InterModuleDCEPass : public InterModuleDCEBase<InterModuleDCEPass> {
  void runOnOperation() override;
  void removeUnusedModulePorts(FModuleOp module,
                               InstanceGraphNode *instanceGraphNode);

  void markAlive(Value value) {
    if (liveSet.count(value))
      return;
    liveSet.insert(value);
    worklist.push_back(value);
  }
  bool isKnownAlive(Value value) const {
    assert(value);
    return liveSet.count(value);
  }
  bool isAssumedDead(Value value) const {
    assert(value);
    return !liveSet.count(value);
  }
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }
  void visitOperation(Operation *op);
  void visitValue(Value value);
  void visitWireOrReg(Operation *op);
  void visitConnect(FConnectLike connect);
  void markBlockExecutable(Block *block);
  void markWireOrReg(Operation *op);
  void markMemOp(MemOp op);
  void markInstanceOp(InstanceOp instanceOp);
  void markUnknownSideEffectOp(Operation *op);
  void rewriteModule(FModuleOp module);

  /// If true, the pass will remove unused ports even if they have carry a
  /// symbol or annotations. This is likely to break the IR, but may be useful
  /// for `circt-reduce` where preserving functional correctness of the IR is
  /// not important.
  bool ignoreDontTouch = false;

private:
  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;
  InstanceGraph *instanceGraph;

  /// A worklist of values whose liveness recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> worklist;
  llvm::DenseSet<Value> liveSet;
};
} // namespace

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrRegOrNode(Operation *op) {
  if (auto name = dyn_cast<FNamableOp>(op))
    if (!name.hasDroppableName())
      return false;
  return !hasDontTouch(op);
}

void InterModuleDCEPass::markWireOrReg(Operation *op) {
  // If the wire/reg has a non-ground type, mark alive for now.
  auto resultValue = op->getResult(0);

  if (!isDeletableWireOrRegOrNode(op))
    markAlive(resultValue);
}

void InterModuleDCEPass::markMemOp(MemOp mem) {
  for (auto result : mem.getResults())
    for (auto user : result.getUsers()) {
      if (auto subfield = dyn_cast<SubfieldOp>(user))
        markAlive(subfield);
    }
}

void InterModuleDCEPass::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, we pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
}

void InterModuleDCEPass::visitOperation(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);
}

void InterModuleDCEPass::markInstanceOp(InstanceOp instance) {
  // Get the module being reference or a null pointer if this is an extmodule.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any results and inouts are
  // overdefined.
  if (!isa<FModuleOp>(op)) {
    auto module = dyn_cast<FModuleLike>(op);
    for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
         ++resultNo) {
      auto portVal = instance.getResult(resultNo);
      // If this is an output to the extmodule, we can ignore it.
      if (module.getPortDirection(resultNo) == Direction::Out)
        continue;

      // Otherwise this is a result from it or an inout, mark it as overdefined.
      markAlive(portVal);
    }
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBody());

  // Ok, it is a normal internal module reference.  Populate
  // resultPortToInstanceResultMapping, and forward liveness.
  for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
       ++resultNo) {
    auto instancePortVal = instance.getResult(resultNo);

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    // Mark don't touch results as alive.
    if (hasDontTouch(modulePortVal)) {
      markAlive(modulePortVal);
      markAlive(instancePortVal);
    }

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);
  }
}

void InterModuleDCEPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  for (auto &op : *block) {
    // Handle each of the special operations in the firrtl dialect.
    if (isWireOrRegOrNode(&op))
      markWireOrReg(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (isa<FConnectLike>(op)) {
      // Nothing to do for connect like op.
      continue;
    } else if (auto mem = dyn_cast<MemOp>(op)) {
      markMemOp(mem);
    } else if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op)) {
      markUnknownSideEffectOp(&op);
    }
  }
}

void InterModuleDCEPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();
  // Mark the ports of public modules as ailve.
  for (auto module : circuit.getBody()->getOps<FModuleOp>()) {
    // FIXME: For now, mark every module as executable.
    markBlockExecutable(module.getBody());
    if (module.isPublic())
      for (auto port : module.getBody()->getArguments())
        markAlive(port);
  }

  // If a value changed liveness then reprocess any of its users.
  while (!worklist.empty()) {
    Value changedVal = worklist.pop_back_val();
    visitValue(changedVal);
    for (Operation *user : changedVal.getUsers())
      visitOperation(user);
  }

  for (auto module : circuit.getBody()->getOps<FModuleOp>())
    removeUnusedModulePorts(module,
                            instanceGraph->lookup(module.moduleNameAttr()));

  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBody()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModule(op); });
}

void InterModuleDCEPass::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  /// Driving input ports propagates the liveness to each instance using the
  // module.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
      markAlive(userOfResultPort);
    return;
  }

  // Driving an instance argument port drives the corresponding argument of the
  // referenced module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = value.cast<mlir::OpResult>();
    // Update the src, when its an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal =
        module.getArgument(instanceResult.getResultNumber());
    return markAlive(modulePortVal);
  }

  if (auto op = value.getDefiningOp())
    for (auto operand : op->getOperands())
      markAlive(operand);
}

void InterModuleDCEPass::visitConnect(FConnectLike connect) {
  // If the dest is dead, then we don't have to propagate liveness.
  if (isAssumedDead(connect.dest()))
    return;

  markAlive(connect.src());
}

void InterModuleDCEPass::rewriteModule(FModuleOp module) {
  auto *body = module.getBody();
  // If a module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!executableBlocks.count(body))
    return;

  // Walk the IR bottom-up when deleting operations.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be dead can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (isAssumedDead(connect.dest())) {
        LLVM_DEBUG(llvm::dbgs() << "DEAD: " << connect << "\n";);
        connect.erase();
      }
      continue;
    }

    if (isWireOrRegOrNode(&op) && isAssumedDead(op.getResult(0))) {
      LLVM_DEBUG(llvm::dbgs() << "DEAD: " << op << "\n";);
      // Users should be already erased.
      assert(op.use_empty() && "no user");
      op.erase();
      continue;
    }

    if (mlir::isOpTriviallyDead(&op))
      op.erase();
  }
}

void InterModuleDCEPass::removeUnusedModulePorts(
    FModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // This tracks port indexes that can be erased.
  SmallVector<unsigned> removalPortIndexes;
  auto ports = module.getPorts();

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  for (auto e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto result = module.getArgument(index);
    assert((!hasDontTouch(result) || isKnownAlive(result)) &&
           "If port has don't touch, it should be know alive");

    if (isKnownAlive(result))
      continue;

    WireOp wire = builder.create<WireOp>(result.getType());
    result.replaceAllUsesWith(wire);
    removalPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.empty())
    return;

  for (auto arg : module.getArguments())
    liveSet.erase(arg);

  // Delete ports from the module.
  module.erasePorts(removalPortIndexes);

  for (auto arg : module.getArguments())
    liveSet.insert(arg);

  LLVM_DEBUG(llvm::for_each(removalPortIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    builder.setInsertionPointToStart(instance->getBlock());
    // Since we will rewrite instance, it is necessary to remove results from
    // liveSet.
    for (auto e : llvm::enumerate(ports)) {
      auto result = instance.getResult(e.index());
      liveSet.erase(result);
    }
    for (auto index : removalPortIndexes) {
      auto result = instance.getResult(index);
      assert(isAssumedDead(result) && "must be dead");
      WireOp wire = builder.create<WireOp>(result.getType());
      result.replaceAllUsesWith(wire);
    }
    // Create a new instance op without unused ports.
    auto newInstance = instance.erasePorts(builder, removalPortIndexes);

    // Mark new results as alive.
    for (auto newResult : newInstance.getResults())
      liveSet.insert(newResult);

    instanceGraph->replaceInstance(instance, newInstance);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.size();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInterModuleDCEPass() {
  return std::make_unique<InterModuleDCEPass>();
}
