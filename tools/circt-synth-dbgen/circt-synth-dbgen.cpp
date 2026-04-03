//===- circt-synth-dbgen.cpp - Synthesis database generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ToolOutputFile.h"

namespace cl = llvm::cl;

using namespace circt;
using namespace circt::synth;
using namespace mlir;

static cl::OptionCategory mainCategory("circt-synth-dbgen Options");

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));
static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));
static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));
static cl::opt<std::string> kind(
    "kind", cl::desc("Predefined database kind to generate: npn"),
    cl::value_desc("name"), cl::init("npn"), cl::cat(mainCategory));
static cl::opt<unsigned> maxInputs(
    "max-inputs", cl::desc("Maximum function input size to generate"),
    cl::init(4), cl::cat(mainCategory));

static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    llvm::errs() << "WARNING: You're attempting to print out a bytecode file.\n"
                    "This is inadvisable as it may cause display problems. If\n"
                    "you REALLY want to taste MLIR bytecode first-hand, you\n"
                    "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::PrettyStackTraceProgram x(argc, argv);

  cl::HideUnrelatedOptions(mainCategory);
  cl::ParseCommandLineOptions(argc, argv, "CIRCT synthesis DB generator\n");

  MLIRContext context;
  context.loadDialect<comb::CombDialect, hw::HWDialect, synth::SynthDialect>();

  auto module = ModuleOp::create(UnknownLoc::get(&context));
  PassManager pm(&context);

  GenPredefinedOptions predefinedOptions;
  predefinedOptions.kind = kind;
  predefinedOptions.maxInputs = maxInputs;
  pm.addPass(createGenPredefined(std::move(predefinedOptions)));
  if (failed(pm.run(module)))
    return 1;

  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(printOp(module, output->os())))
    return 1;
  output->keep();
  return 0;
}
