//===- circt-synth-dbgen.cpp - Synthesis database generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/ExactMIGDatabase.h"
#include "circt/Support/Version.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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
    "kind", cl::desc("Database kind to generate"),
    cl::value_desc("name"), cl::init("mig-exact"), cl::cat(mainCategory));
static cl::opt<unsigned> maxInputs(
    "max-inputs", cl::desc("Maximum function input size to generate"),
    cl::init(4), cl::cat(mainCategory));
static cl::opt<std::string> satSolver(
    "sat-solver", cl::desc("SAT solver backend to use: auto, z3, or cadical"),
    cl::value_desc("backend"), cl::init("auto"), cl::cat(mainCategory));
static cl::opt<int64_t> conflictLimit(
    "conflict-limit",
    cl::desc("Per-SAT-call conflict budget. -1 disables the limit"),
    cl::init(100), cl::cat(mainCategory));

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

  if (kind != "mig-exact" && kind != "MIG_EXACT") {
    llvm::errs() << "unsupported database kind '" << kind << "'\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<hw::HWDialect, synth::SynthDialect>();

  auto module = ModuleOp::create(UnknownLoc::get(&context));
  ExactMIGDatabaseGenOptions options;
  options.maxInputs = maxInputs;
  options.satSolver = satSolver;
  options.conflictLimit = conflictLimit;
  if (failed(emitExactMIGDatabase(module, options)))
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
