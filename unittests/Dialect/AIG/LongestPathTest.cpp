//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace aig;

namespace {

const char *ir = R"MLIR(
    hw.module private @basic(in %clock : !seq.clock, in %a : i2, in %b : i2, out x : i2, out y : i2) {
      %p = seq.firreg %a clock %clock : i2
      %q = seq.firreg %b clock %clock : i2
      %r = hw.instance "inst" @child(a: %p: i2, b: %q: i2) -> (x: i2)
      %s = aig.and_inv not %p, %q, %r : i2
      hw.output %s : i2
    } 
    hw.module private @child(in %a : i2, in %b : i2, out x : i2) {
      %r = aig.and_inv not %a, %b : i2
      hw.output %r : i2
    }
    )MLIR";

TEST(LongestPathTest, ModuleLevel) {
  MLIRContext context;
  context.loadDialect<AIGDialect>();
  Location loc(UnknownLoc::get(&context));
  auto moduleOp = ModuleOp::create(loc);
  OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  // auto attr = ImmediateAttr::get(&context, APInt(12, 0));
  // auto *op0 = context.getLoadedDialect<AIGDialect>()->materializeConstant(
  //     builder, attr, attr.getType(), loc);
  // auto *op1 = context.getLoadedDialect<AIGDialect>()->materializeConstant(
  //     builder, attr, ImmediateType::get(&context, 2), loc);

  // ASSERT_TRUE(op0 && isa<ConstantOp>(op0));
  // ASSERT_EQ(op1, nullptr);
}

} // namespace
