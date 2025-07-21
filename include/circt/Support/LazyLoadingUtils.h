//===- LazyLoadingUtils.h - CIRCT lazy loading common functions--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to help with bytecode lazy loading.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PARSINGUTILS_H
#define CIRCT_SUPPORT_PARSINGUTILS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

namespace circt {
namespace parsing_util {


//===----------------------------------------------------------------------===//
// Initializer lists
//===----------------------------------------------------------------------===//

/// Parses an initializer.
/// An initializer list is a list of operands, types and names on the format:
///  (%arg = %input : type, ...)
ParseResult parseInitializerList(
    mlir::OpAsmParser &parser,
    llvm::SmallVector<mlir::OpAsmParser::Argument> &inputArguments,
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> &inputOperands,
    llvm::SmallVector<Type> &inputTypes, ArrayAttr &inputNames);

// Prints an initializer list.
void printInitializerList(OpAsmPrinter &p, ValueRange ins,
                          ArrayRef<BlockArgument> args);

} // namespace parsing_util
} // namespace circt

#endif // CIRCT_SUPPORT_PARSINGUTILS_H
