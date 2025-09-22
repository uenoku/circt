//===- NetworkConversion.h - Mockturtle Integration -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides high-level interfaces for applying mockturtle algorithms
// to CIRCT IR. It handles the conversion between MLIR and mockturtle networks.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_NETWORKCONVERSION_H
#define LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_NETWORKCONVERSION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <variant>

// Forward declarations
namespace mlir {
class Block;
class SymbolTable;
} // namespace mlir

namespace mockturtle {
class aig_network;
class mig_network;
template <typename Ntk>
class cell_view;
class block_network;
} // namespace mockturtle

namespace circt {
namespace synth {
namespace mockturtle_integration {

/// Variant type representing different network types
using Ntk = std::variant<mockturtle::aig_network *, mockturtle::mig_network *>;

//===----------------------------------------------------------------------===//
// High-level transformation interfaces
//===----------------------------------------------------------------------===//

/// Apply a transformation to an AIG network extracted from the block.
/// The network is constructed from and-inverter operations, transformed
/// in-place, and then written back to the block.
llvm::LogicalResult runAIGNetworkTransforms(
    mlir::Block *block,
    llvm::function_ref<llvm::LogicalResult(mockturtle::aig_network &)>
        transform);

/// Apply a transformation to a MIG network extracted from the block.
llvm::LogicalResult runMIGNetworkTransforms(
    mlir::Block *block,
    llvm::function_ref<llvm::LogicalResult(mockturtle::mig_network &)>
        transform);

/// Apply a transformation that maps a network to cell-view representation.
/// This is used for technology mapping where the result is a set of
/// instantiated cells.
llvm::LogicalResult runAIGNetworkToCellView(
    const mlir::SymbolTable &symbolTable, mlir::Block *block,
    llvm::function_ref<
        mlir::FailureOr<mockturtle::cell_view<mockturtle::block_network>>(
            mockturtle::aig_network &)>
        mapFunction);

/// Apply a generic transformation to a network (auto-detects network type)
llvm::LogicalResult
runNetworkTransforms(mlir::Block *block,
                     llvm::function_ref<llvm::LogicalResult(Ntk &)> transform);

//===----------------------------------------------------------------------===//
// Algorithm Bindings
//===----------------------------------------------------------------------===//

/// Run SOP refactoring on the network.
llvm::LogicalResult runSOPRefactoring(Ntk ntk);

/// Run functional reduction to identify and remove redundant nodes.
llvm::LogicalResult runFunctionalReduction(Ntk ntk);

/// Run MIG algebraic depth rewriting to optimize circuit depth
llvm::LogicalResult runMIGAlgebraicRewriteDepth(mockturtle::mig_network &ntk);

/// Run SOP balancing to optimize circuit structure
llvm::LogicalResult runSOPBalancing(Ntk ntk);

} // namespace mockturtle_integration
} // namespace synth
} // namespace circt

#endif // LIB_DIALECT_SYNTH_TRANSFORMS_MOCKTURTLEINTEGRATION_NETWORKCONVERSION_H
