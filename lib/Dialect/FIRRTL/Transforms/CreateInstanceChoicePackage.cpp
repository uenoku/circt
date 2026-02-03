//===- CreateInstanceChoicePackage.cpp - Create SV package for options ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass creates a SystemVerilog package containing string parameters
// for each option case defined in the FIRRTL circuit. This enables
// post-Verilog configuration of instance choices.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-create-instance-choice-package"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CREATEINSTANCECHOICEPACKAGE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {

struct CreateInstanceChoicePackagePass
    : public circt::firrtl::impl::CreateInstanceChoicePackageBase<
          CreateInstanceChoicePackagePass> {
  void runOnOperation() override;

private:
  void createPackage(CircuitOp circuit);
};

} // namespace

void CreateInstanceChoicePackagePass::createPackage(CircuitOp circuit) {
  // Collect all options and their cases
  llvm::MapVector<StringAttr, SmallVector<StringAttr>> optionCases;
  
  circuit.walk([&](OptionOp option) {
    SmallVector<StringAttr> cases;
    // Add a "Default" case for the default module
    cases.push_back(StringAttr::get(option.getContext(), "Default"));
    
    // Collect all option cases
    for (auto &op : option.getBody().front()) {
      if (auto optionCase = dyn_cast<OptionCaseOp>(op)) {
        cases.push_back(optionCase.getSymNameAttr());
      }
    }
    
    optionCases[option.getSymNameAttr()] = std::move(cases);
  });

  if (optionCases.empty())
    return;

  // Create the package content
  std::string packageContent = "package InstanceChoicePackage;\n";

  for (auto &[optionName, cases] : optionCases) {
    // Generate comment listing available options
    // e.g., "// Available options: "Default", "FPGA", "ASIC""
    packageContent += "\n  // Available options: ";

    for (size_t i = 0; i < cases.size(); ++i) {
      if (i > 0)
        packageContent += ", ";
      packageContent += "\"" + cases[i].getValue().str() + "\"";
    }
    packageContent += "\n";

    // Generate the default selection parameter
    // e.g., "localparam string Platform = "Default";"
    packageContent += "  localparam string " + optionName.getValue().str() +
                     " = \"Default\";\n";
  }

  packageContent += "endpackage\n";

  // Create sv.verbatim operation for the package
  auto *block = circuit.getBodyBlock();
  auto builder = ImplicitLocOpBuilder::atBlockEnd(circuit.getLoc(), block);

  auto outputFile = hw::OutputFileAttr::getFromFilename(
      builder.getContext(), "InstanceChoicePackage.sv",
      /*excludeFromFileList=*/false);

  auto verbatimOp = builder.create<sv::VerbatimOp>(
      packageContent, ValueRange{}, builder.getArrayAttr({}));
  verbatimOp->setAttr("output_file", outputFile);
}

void CreateInstanceChoicePackagePass::runOnOperation() {
  auto circuit = getOperation();
  createPackage(circuit);
  // TODO: Add OM classes.
}
