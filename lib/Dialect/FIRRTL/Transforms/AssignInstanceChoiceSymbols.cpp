//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the AssignInstanceChoiceSymbols pass, which assigns
// globally unique target symbols to all instance choice operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-assign-instance-choice-symbols"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_ASSIGNINSTANCECHOICESYMBOLS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class AssignInstanceChoiceSymbolsPass
    : public circt::firrtl::impl::AssignInstanceChoiceSymbolsBase<
          AssignInstanceChoiceSymbolsPass> {
public:
  void runOnOperation() override;

private:
  FlatSymbolRefAttr assignSymbol(InstanceChoiceOp op,
                                 CircuitNamespace &circuitNamespace);
};
} // namespace

FlatSymbolRefAttr AssignInstanceChoiceSymbolsPass::assignSymbol(
    InstanceChoiceOp op, CircuitNamespace &circuitNamespace) {
  // Skip if already has a target symbol
  if (op.getTargetSymAttr())
    return nullptr;

  // Get the parent module name
  auto parentModule = op->getParentOfType<FModuleLike>();
  assert(parentModule && "instance choice must be inside a module");

  // Get the option name
  auto optionName = op.getOptionNameAttr();

  // Generate the target symbol name.
  // This is not public API and can be generated in any way as long as it's
  // unique.
  SmallString<128> targetSymName;
  {
    llvm::raw_svector_ostream os(targetSymName);
    os << "__target_" << optionName.getValue() << "_"
       << parentModule.getModuleName() << "_" << op.getInstanceName();
  }

  // Ensure global uniqueness using CircuitNamespace
  auto uniqueName =
      StringAttr::get(op.getContext(), circuitNamespace.newName(targetSymName));
  auto targetSym = FlatSymbolRefAttr::get(uniqueName);
  op.setTargetSymAttr(targetSym);

  LLVM_DEBUG(llvm::dbgs() << "Assigned target symbol '" << uniqueName
                          << "' to instance choice '" << op.getInstanceName()
                          << "' in module '" << parentModule.getModuleName()
                          << "'\n");

  return targetSym;
}

void AssignInstanceChoiceSymbolsPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  // Create a circuit namespace for global uniqueness
  CircuitNamespace circuitNamespace(circuit);

  OpBuilder builder(circuit.getContext());
  builder.setInsertionPointToStart(circuit.getBodyBlock());

  llvm::DenseSet<StringAttr> createdMacros;
  bool changed = false;

  // Iterate through all instance choices.
  for (auto *node : instanceGraph) {
    auto module = dyn_cast<FModuleLike>(node->getModule().getOperation());
    if (!module)
      continue;

    for (auto *record : *node) {
      if (auto op = record->getInstance<InstanceChoiceOp>()) {
        auto targetSym = assignSymbol(op, circuitNamespace);
        if (!targetSym)
          continue;
        changed = true;
        // Create macro declaration only if we haven't created it yet
        if (createdMacros.insert(targetSym.getAttr()).second)
          sv::MacroDeclOp::create(builder, circuit.getLoc(),
                                  targetSym.getAttr());
      }
    }
  }
  if (!changed)
    return markAllAnalysesPreserved();

  markAnalysesPreserved<InstanceGraph, InstanceInfo>();
}
