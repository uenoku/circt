//===- Bind.cpp - Mockturtle Algorithm Bindings ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mockturtle algorithm bindings that require RTTI and
// exception handling. These are isolated from the main codebase to avoid
// enabling exceptions globally.
//
//===----------------------------------------------------------------------===//

#include "../NetworkConversion.h"
#include "mockturtle/algorithms/balancing.hpp"
#include "mockturtle/algorithms/balancing/sop_balancing.hpp"
#include "mockturtle/algorithms/functional_reduction.hpp"
#include "mockturtle/algorithms/lut_mapper.hpp"
#include "mockturtle/algorithms/mig_algebraic_rewriting.hpp"
#include "mockturtle/algorithms/node_resynthesis/sop_factoring.hpp"
#include "mockturtle/algorithms/refactoring.hpp"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/mig.hpp"
#include "mockturtle/views/depth_view.hpp"
#include "llvm/Support/LogicalResult.h"
#include <type_traits>
#include <variant>

static llvm::LogicalResult
catchException(const std::function<llvm::LogicalResult()> &func) {
  try {
    return func();
  } catch (const std::exception &e) {
    return llvm::failure();
  }
}

llvm::LogicalResult
circt::synth::mockturtle_integration::runSOPRefactoring(Ntk ntk) {
  return catchException([&]() {
    std::visit(
        [](auto network) {
          using NetworkType = std::remove_pointer_t<decltype(network)>;
          mockturtle::sop_factoring<NetworkType> sop;
          mockturtle::refactoring(*network, sop);
        },
        ntk);
    return llvm::success();
  });
}

llvm::LogicalResult
circt::synth::mockturtle_integration::runFunctionalReduction(Ntk ntk) {
  return catchException([&]() {
    std::visit([](auto network) { mockturtle::functional_reduction(*network); },
               ntk);
    return llvm::success();
  });
}

llvm::LogicalResult
circt::synth::mockturtle_integration::runMIGAlgebraicRewriteDepth(
    mockturtle::mig_network &ntk) {
  return catchException([&]() {
    mockturtle::mig_algebraic_depth_rewriting_params ps;
    ps.strategy = mockturtle::mig_algebraic_depth_rewriting_params::dfs;
    mockturtle::depth_view<mockturtle::mig_network> depthView{ntk};
    mockturtle::mig_algebraic_depth_rewriting(depthView, ps);
    return llvm::success();
  });
}

llvm::LogicalResult
circt::synth::mockturtle_integration::runSOPBalancing(Ntk ntk) {
  return catchException([&]() {
    std::visit(
        [&](auto network) {
          *network = std::move(mockturtle::sop_balancing(*network));
        },
        ntk);
    return llvm::success();
  });
}
