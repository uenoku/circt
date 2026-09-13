//===- StripAssertLike.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace verif {
#define GEN_PASS_DEF_STRIPASSERTLIKEPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace verif;

namespace {
struct StripAssertLikePass
    : public verif::impl::StripAssertLikePassBase<StripAssertLikePass> {
  void runOnOperation() override {
    // Assert-like ops model verification intent only and produce no values,
    // so erasing them is always safe. Their predicate operands go dead and
    // are cleaned up by canonicalization.
    getOperation()->walk([](Operation *op) {
      if (isa<AssertOp, AssumeOp, CoverOp, ClockedAssertOp, ClockedAssumeOp,
               ClockedCoverOp>(op))
        op->erase();
    });
  }
};
} // namespace
