//===- TechLibraries.cpp - Built-in tech libraries --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides built-in technology libraries for the synth tech mapper.
// The library data was generated from the mockturtle genlib files in
// https://github.com/lsils/mockturtle/tree/master/experiments/cell_libraries.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/TechLibraries.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringSwitch.h"
#include <optional>

using namespace circt;
using namespace circt::synth;
using namespace mlir;

static constexpr llvm::StringLiteral sky130TechLibrary =
#include "Sky130TechLibrary.inc"
    ;

static constexpr llvm::StringLiteral asap7TechLibrary =
#include "Asap7TechLibrary.inc"
    ;

static std::optional<llvm::StringRef>
getBuiltinTechLibrarySource(llvm::StringRef libraryName) {
  return llvm::StringSwitch<std::optional<llvm::StringRef>>(libraryName)
      .Case("sky130", llvm::StringRef(sky130TechLibrary))
      .Case("asap7", llvm::StringRef(asap7TechLibrary))
      .Default(std::nullopt);
}

LogicalResult
circt::synth::appendBuiltinTechLibrary(ModuleOp module,
                                       llvm::StringRef libraryName) {
  auto source = getBuiltinTechLibrarySource(libraryName);
  if (!source)
    return module.emitError("unknown built-in tech library '")
           << libraryName << "'; expected one of: asap7, sky130";

  OwningOpRef<ModuleOp> libraryModule =
      parseSourceString<ModuleOp>(*source, module.getContext());
  if (!libraryModule)
    return module.emitError("failed to parse built-in tech library '")
           << libraryName << "'";

  SymbolTable symbolTable(module);
  for (Operation &op : libraryModule->getBody()->getOperations()) {
    auto name = SymbolTable::getSymbolName(&op);
    if (name && symbolTable.lookup(name))
      return module.emitError("built-in tech library '")
             << libraryName << "' conflicts with existing symbol '" << name
             << "'";
  }

  module.getBody()->getOperations().splice(
      module.getBody()->end(), libraryModule->getBody()->getOperations());
  return success();
}
