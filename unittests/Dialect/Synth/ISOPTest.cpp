//===- ISOPTest.cpp - ISOP algorithm unit tests --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the ISOP (Irredundant Sum-of-Products) extraction algorithm.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/SynthUtils.h"
#include "llvm/ADT/APInt.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace circt::synth;

namespace {

/// Test ISOP extraction for various Boolean functions.
/// Verifies both correctness (truth table matches) and irredundancy.
TEST(ISOPTest, SimpleAND) {
  // AND function: f(a,b) = a * b
  // Truth table: 0001 (only true when both inputs are 1)
  llvm::APInt truthTable(4, 0b0001);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  // Verify correctness: computed truth table matches original
  llvm::APInt computed = sop.computeTruthTable();
  EXPECT_EQ(computed, truthTable) << "ISOP truth table doesn't match original";

  // Verify irredundancy: no cube can be removed
  EXPECT_TRUE(sop.isIrredundant()) << "ISOP is not irredundant";

  // AND should produce exactly one cube: (a * b)
  EXPECT_EQ(sop.cubes.size(), 1u);
  EXPECT_EQ(sop.cubes[0].size(), 2u);
}

TEST(ISOPTest, SimpleOR) {
  // OR function: f(a,b) = a + b
  // Truth table: 0111 (true when at least one input is 1)
  llvm::APInt truthTable(4, 0b0111);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());

  // OR should produce two cubes: (a) + (b)
  EXPECT_EQ(sop.cubes.size(), 2u);
}

TEST(ISOPTest, SimpleXOR) {
  // XOR function: f(a,b) = a ^ b
  // Truth table: 0110 (true when inputs differ)
  llvm::APInt truthTable(4, 0b0110);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());

  // XOR should produce two cubes: (a * !b) + (!a * b)
  EXPECT_EQ(sop.cubes.size(), 2u);
  for (const auto &cube : sop.cubes) {
    EXPECT_EQ(cube.size(), 2u);
  }
}

TEST(ISOPTest, Majority3) {
  // MAJ3 function: f(a,b,c) = (a*b) + (a*c) + (b*c)
  // Truth table: 00010111 (true when at least 2 inputs are 1)
  llvm::APInt truthTable(8, 0b11101000); // Note: LSB is minterm 000

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());

  // MAJ3 should produce three cubes: (a*b) + (a*c) + (b*c)
  EXPECT_EQ(sop.cubes.size(), 3u);
  for (const auto &cube : sop.cubes) {
    EXPECT_EQ(cube.size(), 2u);
  }
}

TEST(ISOPTest, ConstantZero) {
  // Constant 0 function
  llvm::APInt truthTable(4, 0);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());

  // Constant 0 should produce no cubes
  EXPECT_EQ(sop.cubes.size(), 0u);
}

TEST(ISOPTest, ConstantOne) {
  // Constant 1 function
  llvm::APInt truthTable(4, 0b1111);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());

  // Constant 1 should produce one empty cube
  EXPECT_EQ(sop.cubes.size(), 1u);
  EXPECT_EQ(sop.cubes[0].size(), 0u);
}

TEST(ISOPTest, ComplexFunction) {
  // Complex 3-input function: f(a,b,c) = a*b + !a*c
  // Truth table: 10111100
  llvm::APInt truthTable(8, 0b00111101);

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());
}

TEST(ISOPTest, FourInputFunction) {
  // 4-input function to test scalability
  // f(a,b,c,d) = a*b*c + c*d
  llvm::APInt truthTable(16, 0);
  for (unsigned i = 0; i < 16; ++i) {
    bool a = (i >> 0) & 1;
    bool b = (i >> 1) & 1;
    bool c = (i >> 2) & 1;
    bool d = (i >> 3) & 1;
    if ((a && b && c) || (c && d))
      truthTable.setBit(i);
  }

  SOPForm sop = extractSOPFromTruthTable(truthTable);

  EXPECT_EQ(sop.computeTruthTable(), truthTable);
  EXPECT_TRUE(sop.isIrredundant());
}

} // namespace
