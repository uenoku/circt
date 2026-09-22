//===- TimingDesignBuilderTest.cpp - Graph builder tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Timing/TimingDesignBuilder.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace circt::synth::timing;

namespace {

// Mapped hierarchy: two leaf cells under a shared submodule instantiated
// twice, plus constants and bit plumbing at the top.
const char *hierIR = R"MLIR(
    hw.module @and2(in %a : i1, in %b : i1, out y : i1) {
      %0 = comb.and %a, %b : i1
      hw.output %0 : i1
    }
    hw.module @inv(in %a : i1, out y : i1) {
      %c = hw.constant true
      %0 = comb.xor %a, %c : i1
      hw.output %0 : i1
    }
    hw.module private @sub(in %x : i1, in %y : i1, out z : i1) {
      %a = hw.instance "g0" @and2(a: %x: i1, b: %y: i1) -> (y: i1)
      %b = hw.instance "g1" @inv(a: %a: i1) -> (y: i1)
      hw.output %b : i1
    }
    hw.module @top(in %a : i1, in %b : i1, out z0 : i1, out z1 : i1) {
      %c = hw.constant false
      %x = comb.extract %a from 0 : (i1) -> i1
      %s0 = hw.instance "u0" @sub(x: %x: i1, y: %b: i1) -> (z: i1)
      %s1 = hw.instance "u1" @sub(x: %x: i1, y: %c: i1) -> (z: i1)
      %cat = comb.concat %s0, %s1 : i1, i1
      %lo = comb.extract %cat from 0 : (i2) -> i1
      %hi = comb.extract %cat from 1 : (i2) -> i1
      hw.output %lo, %hi : i1, i1
    }
    )MLIR";

const char *logicIR = R"MLIR(
    hw.module @top(in %a : i1, in %b : i1, out y : i1) {
      %0 = comb.and %a, %b : i1
      hw.output %0 : i1
    }
    )MLIR";

const char *blackboxIR = R"MLIR(
    hw.module.extern @sram(out dout : i8)
    hw.module @top(in %addr : i4, out dout : i8) {
      %d = hw.instance "m" @sram() -> (dout: i8)
      hw.output %d : i8
    }
    )MLIR";

const char *recursiveIR = R"MLIR(
    hw.module @top(in %a : i1, out y : i1) {
      %y = hw.instance "self" @top(a: %a: i1) -> (y: i1)
      hw.output %y : i1
    }
    )MLIR";

OwningOpRef<ModuleOp> parse(MLIRContext &ctx, const char *ir) {
  ctx.loadDialect<circt::synth::SynthDialect>();
  ctx.loadDialect<hw::HWDialect>();
  ctx.loadDialect<seq::SeqDialect>();
  ctx.loadDialect<comb::CombDialect>();
  return parseSourceString<ModuleOp>(ir, &ctx);
}

DenseSet<StringAttr> libSet(MLIRContext &ctx, ArrayRef<const char *> names) {
  DenseSet<StringAttr> set;
  for (auto name : names)
    set.insert(StringAttr::get(&ctx, name));
  return set;
}

TEST(TimingDesignBuilderTest, HierarchyWithMultiplicity) {
  MLIRContext ctx;
  auto module = parse(ctx, hierIR);
  ASSERT_TRUE(module);
  auto lib = libSet(ctx, {"and2", "inv"});
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto design = builder.build(*module);
  ASSERT_TRUE(succeeded(design));
  // Two sub instances x (1 and2 + 1 inv each) = 4 leaf cells.
  EXPECT_EQ(design->cells.size(), 4u);
  // Instance paths distinguish the copies.
  unsigned u0cells = 0, u1cells = 0;
  for (const auto &cell : design->cells) {
    ASSERT_FALSE(cell.instance.instancePath.empty());
    if (cell.instance.instancePath[0].getValue() == "u0")
      ++u0cells;
    else if (cell.instance.instancePath[0].getValue() == "u1")
      ++u1cells;
    else
      FAIL() << "unexpected path";
  }
  EXPECT_EQ(u0cells, 2u);
  EXPECT_EQ(u1cells, 2u);
  // Pins carry Liberty-ready identity.
  for (const auto &cell : design->cells) {
    EXPECT_FALSE(cell.cellType.getValue().empty());
    for (const auto &pin : cell.pins)
      EXPECT_FALSE(pin.portName.getValue().empty());
  }
  // Nets are singly driven; undriven non-constant nets must be top
  // inputs (primary sources have no driver by construction).
  // Top inputs a, b produce 2 input nets (1 bit each).
  unsigned undriven = 0;
  for (const auto &net : design->nets) {
    if (!net.isConstant && !net.driver.has_value())
      ++undriven;
  }
  EXPECT_EQ(undriven, 2u);
  // Top ports recorded in order: a, b inputs; z0, z1 outputs.
  size_t inputs = 0, outputs = 0;
  for (const auto &port : design->ports) {
    if (port.isInput)
      ++inputs;
    else
      ++outputs;
  }
  EXPECT_EQ(inputs, 2u);
  EXPECT_EQ(outputs, 2u);
}

TEST(TimingDesignBuilderTest, RejectsLogic) {
  MLIRContext ctx;
  auto module = parse(ctx, logicIR);
  ASSERT_TRUE(module);
  auto lib = libSet(ctx, {});
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  EXPECT_TRUE(failed(builder.build(*module)));
  EXPECT_FALSE(builder.getDiagnostics().unsupported.empty());
}

TEST(TimingDesignBuilderTest, RejectsBlackbox) {
  MLIRContext ctx;
  auto module = parse(ctx, blackboxIR);
  ASSERT_TRUE(module);
  auto lib = libSet(ctx, {});
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  EXPECT_TRUE(failed(builder.build(*module)));
  EXPECT_FALSE(builder.getDiagnostics().unsupported.empty());
}

TEST(TimingDesignBuilderTest, RejectsRecursion) {
  MLIRContext ctx;
  auto module = parse(ctx, recursiveIR);
  ASSERT_TRUE(module);
  auto lib = libSet(ctx, {});
  TimingDesignBuilder builder(&ctx, StringAttr::get(&ctx, "top"), lib);
  EXPECT_TRUE(failed(builder.build(*module)));
  EXPECT_FALSE(builder.getDiagnostics().unsupported.empty());
}

TEST(TimingDesignBuilderTest, Deterministic) {
  MLIRContext ctx;
  auto module = parse(ctx, hierIR);
  ASSERT_TRUE(module);
  auto lib = libSet(ctx, {"and2", "inv"});
  auto summarize = [&](TimingDesign &design) {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << design.cells.size() << "/" << design.nets.size() << "/";
    for (const auto &cell : design.cells) {
      os << cell.cellType.getValue() << ":";
      for (const auto &part : cell.instance.instancePath)
        os << part.getValue() << ".";
      os << ";";
    }
    os.flush();
    return out;
  };
  TimingDesignBuilder b1(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto d1 = b1.build(*module);
  ASSERT_TRUE(succeeded(d1));
  TimingDesignBuilder b2(&ctx, StringAttr::get(&ctx, "top"), lib);
  auto d2 = b2.build(*module);
  ASSERT_TRUE(succeeded(d2));
  EXPECT_EQ(summarize(*d1), summarize(*d2));
}

} // namespace
