//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PopulateInstanceChoiceSymbols pass, which populates
// globally unique instance macros for all instance choice operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#define DEBUG_TYPE "firrtl-populate-instance-choice-symbols"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_POPULATEINSTANCECHOICESYMBOLS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

void getOptionCaseMacroName(StringAttr optionName, StringAttr caseName,
                            SmallVectorImpl<char> &macroName) {
  llvm::raw_svector_ostream os(macroName);
  os << "__option_" << optionName.getValue() << "_" << caseName.getValue();
}

class PopulateInstanceChoiceSymbolsPass
    : public impl::PopulateInstanceChoiceSymbolsBase<
          PopulateInstanceChoiceSymbolsPass> {
public:
  void runOnOperation() override;

private:
  /// Assign a unique instance macro symbol to the given instance choice
  /// operation. Returns the assigned symbol, or nullptr if the operation
  /// already has a symbol.
  FlatSymbolRefAttr assignSymbol(InstanceChoiceOp op);

  /// The namespace associated with the circuit.  This is lazily constructed
  /// using `getNamespace`.
  std::optional<CircuitNamespace> circuitNamespace;
  CircuitNamespace &getNamespace() {
    if (!circuitNamespace)
      circuitNamespace = CircuitNamespace(getOperation());
    return *circuitNamespace;
  }
};
} // namespace

FlatSymbolRefAttr
PopulateInstanceChoiceSymbolsPass::assignSymbol(InstanceChoiceOp op) {
  // Skip if already has an instance macro.
  if (op.getInstanceMacroAttr())
    return nullptr;

  // Get the parent module name.
  auto parentModule = op->getParentOfType<FModuleLike>();

  // Get the option name.
  auto optionName = op.getOptionNameAttr();

  // Generate the instance macro name.
  // This is not public API and can be generated in any way as long as it's
  // unique.
  SmallString<128> instanceMacroName;
  {
    llvm::raw_svector_ostream os(instanceMacroName);
    os << "__target_" << optionName.getValue() << "_"
       << parentModule.getModuleName() << "_" << op.getInstanceName();
  }

  // Ensure global uniqueness using CircuitNamespace.
  auto uniqueName = StringAttr::get(op.getContext(),
                                    getNamespace().newName(instanceMacroName));
  auto instanceMacro = FlatSymbolRefAttr::get(uniqueName);
  op.setInstanceMacroAttr(instanceMacro);

  LLVM_DEBUG(llvm::dbgs() << "Assigned instance macro '" << uniqueName
                          << "' to instance choice '" << op.getInstanceName()
                          << "' in module '" << parentModule.getModuleName()
                          << "'\n");

  return instanceMacro;
}

void PopulateInstanceChoiceSymbolsPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  OpBuilder builder(circuit.getContext());
  builder.setInsertionPointToStart(circuit.getBodyBlock());

  llvm::DenseSet<StringAttr> createdInstanceMacros;
  bool changed = false;

  llvm::MapVector<StringAttr, std::pair<llvm::SmallVector<Attribute>,
                                        llvm::SmallVector<InstanceChoiceOp>>>
      cases;

  // First, walk all OptionOps and assign case macros to OptionCaseOps.
  for (auto optionOp : circuit.getOps<OptionOp>()) {
    auto optionName = optionOp.getSymNameAttr();

    for (auto caseOp : optionOp.getOps<OptionCaseOp>()) {
      // Skip if already has a case macro.
      if (caseOp.getCaseMacroAttr())
        continue;

      auto caseName = caseOp.getSymNameAttr();
      SmallString<128> caseMacroName;
      getOptionCaseMacroName(optionName, caseName, caseMacroName);

      // Ensure global uniqueness using CircuitNamespace.
      auto caseMacro = FlatSymbolRefAttr::get(
          circuit.getContext(), getNamespace().newName(caseMacroName));

      // Set the case_macro attribute on the OptionCaseOp.
      caseOp.setCaseMacroAttr(caseMacro);
      changed = true;

      // Create macro declaration.
      sv::MacroDeclOp::create(builder, circuit.getLoc(), caseMacro.getValue());

      LLVM_DEBUG(llvm::dbgs() << "Assigned case macro '" << caseMacro.getValue()
                              << "' to option case '" << caseName
                              << "' in option '" << optionName << "'\n");
    }
  }

  // Second, iterate through all instance choices and assign instance macros.
  SmallVector<InstanceChoiceOp> instanceChoices;
  instanceGraph.walkPostOrder([&](igraph::InstanceGraphNode &node) {
    auto module = dyn_cast<FModuleLike>(node.getModule().getOperation());
    if (!module)
      return;

    for (auto *record : node) {
      auto op = record->getInstance<InstanceChoiceOp>();

      if (!op)
        continue;
      instanceChoices.push_back(op);
      cases[op.getOptionNameAttr()].second.push_back(op);
      for (auto caseName : op.getCaseNamesAttr()) {
        cases[op.getOptionNameAttr()].first.push_back(caseName);
      }

      auto instanceMacro = assignSymbol(op);
      if (!instanceMacro)
        continue;
      changed = true;

      // Create instance macro declaration only if we haven't created it
      // yet.
      if (createdInstanceMacros.insert(instanceMacro.getAttr()).second)
        sv::MacroDeclOp::create(builder, circuit.getLoc(),
                                instanceMacro.getAttr());
    }
  });

  // // For each public module generate a header file that enumerate all options.
  // InstancePathCache instancePathCache(instanceGraph);
  // for (auto module : circuit.getOps<FModuleOp>()) {
  //   if (!module.isPublic())
  //     continue;
  //   auto *node = instanceGraph[module];

  //   OpBuilder buffer(module);

  //   // Emit "// Include this file to configure following instances:"
  //   auto emitFile = emit::FileOp::create(
  //       builder, circuit.getLoc(),
  //       "example-targets-" + module.getModuleName() + "-cfg.svh");
  //   builder.setInsertionPointToStart(&emitFile.getBodyRegion().front());
  //   for (auto &[optionName, pair] : cases) {
  //     auto [caseNames, instanceChoices] = pair;
  //     if (instanceChoices.empty())
  //       continue;
  //     emit::VerbatimOp::create(
  //         buffer, circuit.getLoc(),
  //         builder.getStringAttr("// ======== Configure option '" +
  //                               optionName.getValue() + "':\n"));
  //     // List of instances
  //     for (auto instanceChoice : instanceChoices) {
  //       // Get paths.
  //       auto parent = instanceChoice->getParentOfType<FModuleLike>();
  //       auto paths = instancePathCache.getRelativePaths(parent, node);
  //       for (auto path : paths) {
  //         // Construct verilog string for now.
  //         SmallString<64> verilogPath;
  //         for (auto record : path) {
  //           verilogPath.append(record.getInstanceName());
  //           verilogPath.append(".");
  //         }
  //         emit::VerbatimOp::create(
  //             buffer, circuit.getLoc(),
  //             builder.getStringAttr("  // " + verilogPath.str() + "\n"));
  //       }
  //     }
  //     // Include examples.
  //     for (auto caseName : caseNames)
  //       emit::VerbatimOp::create(
  //           buffer, circuit.getLoc(),
  //           builder.getStringAttr(
  //               "//   `include \"targets-" + module.getModuleName() + "-" +
  //               optionName.getValue() + ".svh\"\n"));
  //     emit::VerbatimOp::create(buffer, circuit.getLoc(),
  //                              builder.getStringAttr("// ========\n\n"));
  //   }
  // }

  circuitNamespace.reset();
  if (!changed)
    return markAllAnalysesPreserved();

  markAnalysesPreserved<InstanceGraph, InstanceInfo>();
}
