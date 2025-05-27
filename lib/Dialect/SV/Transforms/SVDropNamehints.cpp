//===- HWCleanup.cpp - HW Cleanup Pass ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs various cleanups and canonicalization
// transformations for hw.module bodies.  This is intended to be used early in
// the HW/SV pipeline to expose optimization opportunities that require global
// analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace sv {
#define GEN_PASS_DEF_SVDROPNAMEHINTS
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {
struct SVDropNamehintsPass
    : public circt::sv::impl::SVDropNamehintsBase<SVDropNamehintsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void SVDropNamehintsPass::runOnOperation() {
  auto operation = getOperation();
  operation->walk([&](Operation *op) { op->removeAttr("sv.namehint"); });
}