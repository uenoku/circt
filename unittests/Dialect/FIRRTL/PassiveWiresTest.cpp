//===- PassiveWiresTest.cpp - Passive wire analysis preservation tests
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt::firrtl;

namespace {

struct CountingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingAnalysis)

  static unsigned constructions;
  explicit CountingAnalysis(Operation *) { ++constructions; }
};
unsigned CountingAnalysis::constructions = 0;

struct CacheAnalysisPass
    : PassWrapper<CacheAnalysisPass, OperationPass<FModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CacheAnalysisPass)

  void runOnOperation() override {
    getAnalysis<CountingAnalysis>();
    markAllAnalysesPreserved();
  }
};

void checkAnalysisPreservation(StringRef source,
                               unsigned expectedConstructions) {
  MLIRContext context;
  context.loadDialect<FIRRTLDialect>();
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);

  CountingAnalysis::constructions = 0;
  PassManager pm(&context);
  auto &modulePM = pm.nest<CircuitOp>().nest<FModuleOp>();
  modulePM.addPass(std::make_unique<CacheAnalysisPass>());
  modulePM.addPass(createPassiveWires());
  modulePM.addPass(std::make_unique<CacheAnalysisPass>());
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_EQ(CountingAnalysis::constructions, expectedConstructions);
}

TEST(PassiveWiresTest, PreservesAnalysesWithoutChanges) {
  checkAnalysisPreservation(R"mlir(
    firrtl.circuit "Top" {
      firrtl.module @Top(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
        %w = firrtl.wire : !firrtl.uint<1>
        firrtl.matchingconnect %w, %a : !firrtl.uint<1>
        firrtl.matchingconnect %b, %w : !firrtl.uint<1>
      }
    }
  )mlir",
                            1);
}

TEST(PassiveWiresTest, InvalidatesAnalysesAfterChangingWireTypes) {
  checkAnalysisPreservation(R"mlir(
    firrtl.circuit "Top" {
      firrtl.module @Top() {
        %w = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
      }
    }
  )mlir",
                            2);
}

TEST(PassiveWiresTest, InvalidatesAnalysesAfterExpandingConnects) {
  checkAnalysisPreservation(R"mlir(
    firrtl.circuit "Top" {
      firrtl.module @Top(in %a: !firrtl.bundle<a flip: uint<1>>,
                        out %b: !firrtl.bundle<a flip: uint<1>>) {
        firrtl.connect %b, %a : !firrtl.bundle<a flip: uint<1>>,
                                !firrtl.bundle<a flip: uint<1>>
      }
    }
  )mlir",
                            2);
}

TEST(PassiveWiresTest, PreservesAnalysesInEmptyModule) {
  checkAnalysisPreservation(R"mlir(
    firrtl.circuit "Top" {
      firrtl.module @Top() {}
    }
  )mlir",
                            1);
}

} // namespace
