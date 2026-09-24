//===- CheckDomain.cpp - Verify clock domains ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
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
  return "value";
}

LogicalResult
CheckDomainPass::checkValue(CheckClockDomainOp check, Value value, Value clock,
                            llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return success();

  if (auto clocked = dyn_cast_or_null<Clocked>(value.getDefiningOp())) {
    if (clocked.getClk() != clock) {
      check.emitOpError() << "input depends on a sequential element clocked by "
                          << describeValue(clocked.getClk())
                          << ", not the expected clock "
                          << describeValue(clock);
      return failure();
    }
    return success();
  }

  auto *definingOp = value.getDefiningOp();
  if (!definingOp)
    return success();

  for (Value operand : definingOp->getOperands())
    if (failed(checkValue(check, operand, clock, visited)))
      return failure();
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

  auto module = getOperation().lookupSymbol<hw::HWModuleOp>(moduleName);
  if (!module) {
    emitError(getOperation().getLoc())
        << "could not find HW module '" << moduleName << "'";
    return signalPassFailure();
  }

  // Flatten the selected module's private implementation hierarchy. The
  // selected module remains public, preventing it from being inlined into any
  // of its parents.
  hw::FlattenModulesOptions flattenOptions;
  flattenOptions.inlineWithState = true;
  mlir::OpPassManager pipeline("builtin.module");
  pipeline.addPass(hw::createFlattenModules(flattenOptions));
  pipeline.addNestedPass<hw::HWModuleOp>(mlir::createCanonicalizerPass());
  pipeline.addNestedPass<hw::HWModuleOp>(mlir::createCSEPass());
  if (failed(runPipeline(pipeline, getOperation())))
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
