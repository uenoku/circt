//===- AIGERRunner.cpp - Run external logic solvers on AIGER files --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that runs external logic solvers
// (ABC/Yosys/mockturtle) on AIGER files by exporting the current module to
// AIGER format, running the solver, and importing the results back.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportAIGER.h"
#include "circt/Conversion/ImportAIGER.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::aig;

namespace circt {
namespace aig {
#define GEN_PASS_DEF_AIGERRUNNER
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

namespace {
struct Converter {
  using Object = std::pair<Value, size_t>;
  DenseMap<Value, SmallVector<int>> unknownOperationOperandMap;
  DenseMap<std::pair<Operation *, size_t>, SmallVector<int>>
      unknownOperationResultMap;

  // See. ExportAIGER.cpp
  bool unknownOperationOperandHandler(OpOperand &op, size_t bitPos,
                                      size_t outputIndex) {
    if (unknownOperationOperandMap.find(op.get()) ==
        unknownOperationOperandMap.end())
      unknownOperationOperandMap[op.get()].assign(
          hw::getBitWidth(op.get().getType()), -1);
    unknownOperationOperandMap[op.get()][bitPos] = outputIndex;
    return true;
  }

  // See. ExportAIGER.cpp
  bool unknownOperationResultHandler(OpResult &opResult, size_t bitPos,
                                     size_t inputIndex) {
    auto *op = opResult.getOwner();
    auto index = opResult.getResultNumber();
    if (unknownOperationResultMap.find(std::make_pair(op, index)) ==
        unknownOperationResultMap.end())
      unknownOperationResultMap[std::make_pair(op, index)].assign(
          hw::getBitWidth(op->getResult(index).getType()), -1);
    unknownOperationResultMap[std::make_pair(op, index)][bitPos] = inputIndex;
    return true;
  }

  void replaceModule(hw::HWModuleOp module, hw::HWModuleOp replaced) {
    mlir::IRMapping mapping;
    OpBuilder builder(module->getContext());
    builder.setInsertionPointToStart(module.getBodyBlock());
    SmallVector<Value> inputsToReplace;
    // for ()
    // builder.create<hw::InstanceOp>(builder.getUnknownLoc(), replaced.getName());
  }
};
struct AIGERRunnerPass : public impl::AIGERRunnerBase<AIGERRunnerPass> {
  AIGERRunnerPass() = default;
  AIGERRunnerPass(const AIGERRunnerOptions &options) {
    solverPath = options.solverPath;
    solverArgs = options.solverArgs;
  }

  void runOnOperation() override;

private:
  using AIGERRunnerBase::AIGERRunnerBase::solverArgs;
  using AIGERRunnerBase::AIGERRunnerBase::solverPath;

  // Helper methods
  LogicalResult runSolver(StringRef inputPath, StringRef outputPath);
  LogicalResult exportToAIGER(Converter &converter, hw::HWModuleOp module,
                              StringRef outputPath);
  LogicalResult importFromAIGER(Converter &converter, StringRef inputPath,
                                hw::HWModuleOp module);
};

} // namespace

void AIGERRunnerPass::runOnOperation() {
  // Get the module being transformed
  auto module = getOperation();

  // Create temporary files for AIGER input/output
  SmallString<128> tempDir;
  if (auto error =
          llvm::sys::fs::createUniqueDirectory("aiger-runner", tempDir)) {
    module.emitError("failed to create temporary directory: " +
                     error.message());
    return signalPassFailure();
  }

  SmallString<128> inputPath(tempDir);
  llvm::sys::path::append(inputPath, "input.aig");
  SmallString<128> outputPath(tempDir);
  llvm::sys::path::append(outputPath, "output.aig");

  Converter converter;

  // Export current module to AIGER format
  if (failed(
          exportToAIGER(converter, cast<hw::HWModuleOp>(module), inputPath))) {
    module.emitError("failed to export module to AIGER format");
    return signalPassFailure();
  }

  // Run the external solver
  if (failed(runSolver(inputPath, outputPath))) {
    module.emitError("failed to run external solver");
    return signalPassFailure();
  }

  // Import the results back
  if (failed(importFromAIGER(converter, outputPath,
                             cast<hw::HWModuleOp>(module)))) {
    module.emitError("failed to import results from AIGER format");
    return signalPassFailure();
  }

  // Clean up temporary files
  if (llvm::sys::fs::remove(inputPath)) {
    module.emitError("failed to remove input file: " + inputPath.str());
    return signalPassFailure();
  }
  if (llvm::sys::fs::remove(outputPath)) {
    module.emitError("failed to remove output file: " + outputPath.str());
    return signalPassFailure();
  }
  if (llvm::sys::fs::remove(tempDir)) {
    module.emitError("failed to remove temporary directory: " + tempDir.str());
    return signalPassFailure();
  }
}

LogicalResult AIGERRunnerPass::runSolver(StringRef inputPath,
                                         StringRef outputPath) {
  // Prepare command line arguments
  SmallVector<StringRef> args;
  std::vector<std::string> solverArgsStr;

  // Process solver arguments
  for (auto solverArg : solverArgs) {
    std::string arg = solverArg;
    // Replace special tokens with the actual paths
    size_t pos = 0;
    while ((pos = arg.find("inputFile", pos)) != std::string::npos) {
      llvm::errs() << "before: " << arg << "\n";
      arg = arg.replace(pos, 9, inputPath.str());
      llvm::errs() << "after: " << arg << "\n";
      pos += inputPath.size();
    }
    pos = 0;
    while ((pos = arg.find("outputFile", pos)) != std::string::npos) {
      arg = arg.replace(pos, 10, outputPath.str());
      pos += outputPath.size();
    }
    llvm::errs() << "arg: " << arg << "\n";
    solverArgsStr.push_back(arg);
  }
  // Run the solver
  std::string error;
  auto findProgram = llvm::sys::findProgramByName(solverPath);
  if (findProgram.getError()) {
    llvm::errs() << "Failed to find solver program: " << solverPath << "\n";
    return failure();
  }

  llvm::errs() << "solverPath: " << *findProgram << "\n";
  args.push_back(*findProgram);
  for (auto &arg : solverArgsStr) {
    args.push_back(arg);
    llvm::errs() << "arg: " << arg << "\n";
  }

  int result = llvm::sys::ExecuteAndWait(findProgram.get(), args);

  if (result != 0) {
    llvm::errs() << "Solver execution failed with error: " << error << "\n";
    return failure();
  }

  return success();
}

LogicalResult AIGERRunnerPass::exportToAIGER(Converter &converter,
                                             hw::HWModuleOp module,
                                             StringRef outputPath) {
  std::error_code error;
  auto outputFile = mlir::openOutputFile(outputPath);
  if (!outputFile) {
    module.emitError("failed to open output file: " + outputPath.str());
    return failure();
  }

  ExportAIGEROptions options;
  options.binaryFormat = true;
  options.includeSymbolTable = false; // For better performance
  options.includeComments = false;
  options.handleUnknownOperation = true;
  options.unknownOperationOperandHandler = [&converter](OpOperand &op,
                                                        size_t bitPos,
                                                        size_t outputIndex) {
    return converter.unknownOperationOperandHandler(op, bitPos, outputIndex);
  };
  options.unknownOperationResultHandler =
      [&converter](OpResult opResult, size_t bitPos, size_t inputIndex) {
        return converter.unknownOperationResultHandler(opResult, bitPos,
                                                       inputIndex);
      };

  auto result = exportAIGER(module, outputFile->os(), &options);
  outputFile->keep();
  return result;
}

LogicalResult AIGERRunnerPass::importFromAIGER(Converter &converter,
                                               StringRef inputPath,
                                               hw::HWModuleOp module) {
  ImportAIGEROptions options;
  // Open the input file
  llvm::SourceMgr sourceMgr;
  auto inputFile = mlir::openInputFile(inputPath);
  if (!inputFile) {
    module.emitError("failed to open input file: " + inputPath.str());
    return failure();
  }
  sourceMgr.AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  mlir::TimingScope ts;
  mlir::Block block;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(&block);
  auto buffer = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  if (failed(importAIGER(sourceMgr, module->getContext(), ts, buffer))) {
    module.emitError("failed to import module from AIGER format");
    return failure();
  }
  auto newModule = cast<hw::HWModuleOp>(buffer.getBody()->front());

  llvm::errs() << "newModule: " << newModule.getName() << "\n";
  newModule.dump();

  // Ok, let's replace the original module with the imported one.
  converter.replaceModule(module, newModule);

  // Replace the original module with the imported one
  return llvm::success();
}