//===- RemoveUnusedPorts.cpp - Remove Dead Ports ----------------*- C++ -*-===//
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

/// Return true if this is a wire or register.
static bool isWireOrReg(Operation *op) {
  return isa<WireOp>(op) || isa<RegResetOp>(op) || isa<RegOp>(op);
}

namespace {
struct RemoveUnusedPortsPass
    : public RemoveUnusedPortsBase<RemoveUnusedPortsPass> {
  void runOnOperation() override;
  void removeUnusedModulePorts(FModuleOp module,
                               InstanceGraphNode *instanceGraphNode);

  void markAlive(Value value) {
    if (liveSet.count(value))
      return;
    liveSet.insert(value);
    worklist.push_back(value);
  }
  bool isKnownAlive(Value value) { return liveSet.count(value); }
  bool isAssumedDead(Value value) { return !isKnownAlive(value); }
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }
  void visitOperation(Operation *op);
  void visitValue(Value value);
  void visitWireOrReg(Operation *op);
  void visitConnect(FConnectLike connect);
  void markBlockExecutable(Block *block);
  void markWireOrReg(Operation *op);
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
  DenseSet<Value> liveSet;
};
} // namespace

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrRegOrNode(Operation *op) {
  if (auto wire = dyn_cast<FNamableOp>(op))
    if (!wire.hasDroppableName())
      return false;
  return !hasDontTouch(op);
}

void RemoveUnusedPortsPass::markWireOrReg(Operation *op) {
  // If the wire/reg has a non-ground type, mark alive for now.
  auto resultValue = op->getResult(0);

  if (!isDeletableWireOrRegOrNode(op))
    markAlive(resultValue);
}

void RemoveUnusedPortsPass::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, we pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
}

void RemoveUnusedPortsPass::visitOperation(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  // If this is a operation with special handling, handle it specially.
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);

  if (!mlir::MemoryEffectOpInterface::hasNoEffect(op)) {
    // If the op has side effects, then we already marked its results and
    // operands alive.
    return;
  }

  if (llvm::all_of(op->getOperands(),
                   [&](Value value) { return isAssumedDead(value); }))
    return;

  for (auto operand : op->getOperands())
    markAlive(operand);
}

void RemoveUnusedPortsPass::markInstanceOp(InstanceOp instance) {
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

    // Mark don't touch results as overdefined
    if (hasDontTouch(modulePortVal)) {
      markAlive(modulePortVal);
      markAlive(instancePortVal);
    }

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);
  }
}

void RemoveUnusedPortsPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  for (auto &op : *block) {
    // Handle each of the special operations in the firrtl dialect.
    if (isWireOrReg(&op) || isa<NodeOp>(op))
      markWireOrReg(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (isa<FConnectLike>(op)) {
      // Nothing to do for connect like op.
      continue;
    } else if (mlir::MemoryEffectOpInterface::hasNoEffect(&op)) {
      markUnknownSideEffectOp(&op);
    }
  }
}

void RemoveUnusedPortsPass::runOnOperation() {
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();
  // Mark the ports of public modules as ailve.
  for (auto module : circuit.getBody()->getOps<FModuleOp>()) {
    // FIXME: For now, mark every module as executable.
    markBlockExecutable(module.getBody());
    for (auto port : module.getBody()->getArguments())
      markAlive(port);
  }

  // If a value changed liveness then reprocess any of its users.
  while (!worklist.empty()) {
    Value changedVal = worklist.pop_back_val();
    for (Operation *user : changedVal.getUsers()) {
      // if (isBlockExecutable(user->getBlock()))
      visitOperation(user);
    }
    visitValue(changedVal);
  }

  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  // Iterate in the reverse order of instance graph iterator, i.e. from leaves
  // to top.
  for (auto module : circuit.getBody()->getOps<FModuleOp>())
    removeUnusedModulePorts(module,
                            instanceGraph->lookup(module.moduleNameAttr()));

  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBody()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModule(op); });
}

void RemoveUnusedPortsPass::visitValue(Value value) {
  //  if (isKnownAlive(value))
  //    return;
  //
  markAlive(value);

  /// Driving result ports propagates the value to each instance using the
  // module.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
      markAlive(userOfResultPort);
    return;
  }

  auto src = value.cast<mlir::OpResult>();

  // Driving an instance argument port drives the corresponding argument of the
  // referenced module.
  if (auto instance = src.getDefiningOp<InstanceOp>()) {
    // Update the src, when its an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal = module.getArgument(src.getResultNumber());
    return markAlive(modulePortVal);
  }
  // Driving result ports propagates the value to each instance using the
  // module.
  if (auto blockArg = src.dyn_cast<BlockArgument>()) {
    for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
      markAlive(userOfResultPort);
    return;
  }

  if (auto op = value.getDefiningOp()) {
    for (auto operand : op->getOperands())
      markAlive(operand);
  }
}

void RemoveUnusedPortsPass::visitConnect(FConnectLike connect) {
  // If the dest is dead, then we don't have to propagate liveness.
  if (isAssumedDead(connect.dest()))
    return;

  markAlive(connect.src());

  // // Driving result ports propagates the value to each instance using the
  // // module.
  // if (auto blockArg = connect.src().dyn_cast<BlockArgument>()) {
  //   for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
  //     markAlive(userOfResultPort);
  //   return;
  // }

  // auto src = connect.src().cast<mlir::OpResult>();

  // // Driving an instance argument port drives the corresponding argument of
  // the
  // // referenced module.
  // if (auto instance = src.getDefiningOp<InstanceOp>()) {
  //   // Update the src, when its an instance op.
  //   auto module =
  //       dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
  //   if (!module)
  //     return;

  //   BlockArgument modulePortVal = module.getArgument(src.getResultNumber());
  //   return markAlive(modulePortVal);
  // }
}

void RemoveUnusedPortsPass::rewriteModule(FModuleOp module) {
  // module.dump();
  auto *body = module.getBody();
  // If a module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!executableBlocks.count(body))
    return;

  // Walk the IR bottom-up when folding.  We often fold entire chains of
  // operations into constants, which make the intermediate nodes dead.  Going
  // bottom up eliminates the users of the intermediate ops, allowing us to
  // aggressively delete them.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be constant can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (auto *destOp = connect.dest().getDefiningOp()) {
        if (isAssumedDead(connect.dest()))
          connect.erase();
      }
      continue;
    }

    if ((isWireOrReg(&op) || isa<NodeOp>(op)) &&
        isAssumedDead(op.getResult(0))) {
      llvm::dbgs() << "Op is assumed to be dead: " << op << "\n";
      // Users must be already erased.
      assert(op.use_empty());
      op.erase();
    }

    // We only fold single-result ops and instances in practice, because they
    // are the expressions.
    if (isa<InstanceOp>(op))
      continue;

    if (mlir::isOpTriviallyDead(&op))
      op.erase();

    // // We only fold single-result ops and instances in practice, because they
    // // are the expressions.
    // if (op.getNumResults() != 1 && !isa<InstanceOp>(op))
    //   continue;
  }
}

void RemoveUnusedPortsPass::removeUnusedModulePorts(
    FModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // This tracks port indexes that can be erased.
  SmallVector<unsigned> removalPortIndexes;
  // This tracks constant values of output ports. None indicates an invalid
  // value.
  auto ports = module.getPorts();

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  for (auto e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto arg = module.getArgument(index);
    if (isAssumedDead(arg)) {
      auto result = module.getArgument(index);
      SmallString<16> foo;
      foo = "dead_port_";
      foo += e.value().name.getValue();
      WireOp wire = builder.create<WireOp>(result.getType(), foo,
                                           NameKindEnum::DroppableName);
      result.replaceAllUsesWith(wire);
      removalPortIndexes.push_back(index);
    }
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalPortIndexes);
  LLVM_DEBUG(llvm::for_each(removalPortIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    for (auto index : removalPortIndexes) {
      auto result = instance.getResult(index);
      SmallString<16> foo;
      foo += module.getName();
      foo += "_dead_port_";
      foo += std::to_string(index);
      WireOp wire = builder.create<WireOp>(result.getType(), foo,
                                           NameKindEnum::DroppableName);
      result.replaceAllUsesWith(wire);
    }

    // Create a new instance op without unused ports.
    instance.erasePorts(builder, removalPortIndexes);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.size();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createRemoveUnusedPortsPass(bool ignoreDontTouch) {
  auto pass = std::make_unique<RemoveUnusedPortsPass>();
  pass->ignoreDontTouch = ignoreDontTouch;
  return pass;
}
