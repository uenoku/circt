//===- AssignInstanceChoiceSymbols.cpp - Assign symbols to instance choices
//===//
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
    return op.getTargetSymAttr();

  // Get the parent module name
  auto parentModule = op->getParentOfType<FModuleLike>();
  if (!parentModule)
    return nullptr;

  // Get the option name
  auto optionName = op.getOptionNameAttr();

  // Generate the target symbol name
  // Format: __target_<Option>_<module>_<instance>
  SmallString<128> targetSymName;
  {
    llvm::raw_svector_ostream os(targetSymName);
    os << "__target_" << optionName.getValue() << "_"
       << parentModule.getModuleName() << "_" << op.getInstanceName();
  }

  // Ensure global uniqueness using CircuitNamespace
  auto uniqueName = circuitNamespace.newName(targetSymName);
  auto targetSymAttr = StringAttr::get(op.getContext(), uniqueName);
  auto targetSym = FlatSymbolRefAttr::get(targetSymAttr);

  // Set the target symbol attribute
  op.setTargetSymAttr(targetSym);

  LLVM_DEBUG(llvm::dbgs() << "Assigned target symbol '" << uniqueName
                          << "' to instance choice '" << op.getInstanceName()
                          << "' in module '" << parentModule.getModuleName()
                          << "'\n");
}

void AssignInstanceChoiceSymbolsPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Create a circuit namespace for global uniqueness
  CircuitNamespace circuitNamespace(circuit);

  // Add macro declarations for all assigned symbols
  OpBuilder builder(circuit.getContext());
  builder.setInsertionPointToStart(circuit.getBodyBlock());

  // Track which symbols we've already created macro declarations for
  llvm::DenseSet<StringAttr> createdMacros;

  // Iterate through all modules in the instance graph
  for (auto *node : instanceGraph) {
    auto module = dyn_cast<FModuleLike>(node->getModule().getOperation());
    if (!module)
      continue;

    // Find all instance choice operations in this module
    for (auto *record : *node) {
      if (auto op = record->getInstance<InstanceChoiceOp>()) {
        auto targetSym = assignSymbol(op, circuitNamespace);
        assert(targetSym && "expected target symbol to be assigned");
        // Create macro declaration only if we haven't created it yet
        if (!createdMacros.insert(targetSym.getAttr()).second)
          continue;
        builder.create<sv::MacroDeclOp>(circuit.getLoc(), targetSym.getAttr());
      }
    }
  }
}

namespace circt {
namespace firrtl {
std::unique_ptr<mlir::Pass> createAssignInstanceChoiceSymbolsPass() {
  return std::make_unique<AssignInstanceChoiceSymbolsPass>();
}
} // namespace firrtl
} // namespace circt
