//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Support/TruthTable.h"
#include "llvm/ADT/APInt.h"
#include "gtest/gtest.h"

using namespace circt;
using namespace circt::synth;
using namespace llvm;

static APInt buildExpectedExpanded(const APInt &tt,
                                   ArrayRef<unsigned> inputMapping,
                                   unsigned numMergedInputs) {
  unsigned numOrigInputs = inputMapping.size();
  unsigned mergedSize = 1u << numMergedInputs;
  APInt result(mergedSize, 0);
  for (unsigned mergedIdx = 0; mergedIdx < mergedSize; ++mergedIdx) {
    unsigned origIdx = 0;
    for (unsigned i = 0; i < numOrigInputs; ++i) {
      if ((mergedIdx >> inputMapping[i]) & 1)
        origIdx |= (1u << i);
    }
    if (tt[origIdx])
      result.setBit(mergedIdx);
  }
  return result;
}

TEST(CutRewriterTest, ExpandTruthTableIdentity) {
  BinaryTruthTable tt(2, 1);
  tt.setOutput(APInt(2, 0), APInt(1, 0));
  tt.setOutput(APInt(2, 1), APInt(1, 1));
  tt.setOutput(APInt(2, 2), APInt(1, 1));
  tt.setOutput(APInt(2, 3), APInt(1, 0));

  SmallVector<unsigned> mapping = {0, 1};
  APInt expanded =
      circt::detail::expandTruthTableForMergedInputs(tt.table, mapping, 2);
  APInt expected = buildExpectedExpanded(tt.table, mapping, 2);
  EXPECT_EQ(expanded, expected);
}

TEST(CutRewriterTest, ExpandTruthTablePermuted) {
  BinaryTruthTable tt(2, 1);
  tt.setOutput(APInt(2, 0), APInt(1, 0));
  tt.setOutput(APInt(2, 1), APInt(1, 0));
  tt.setOutput(APInt(2, 2), APInt(1, 1));
  tt.setOutput(APInt(2, 3), APInt(1, 0));

  SmallVector<unsigned> mapping = {2, 0};
  APInt expanded =
      circt::detail::expandTruthTableForMergedInputs(tt.table, mapping, 3);
  APInt expected = buildExpectedExpanded(tt.table, mapping, 3);
  EXPECT_EQ(expanded, expected);
}

TEST(CutRewriterTest, ExpandTruthTableLargeMergedInputs) {
  BinaryTruthTable tt(2, 1);
  tt.setOutput(APInt(2, 0), APInt(1, 1));
  tt.setOutput(APInt(2, 1), APInt(1, 0));
  tt.setOutput(APInt(2, 2), APInt(1, 0));
  tt.setOutput(APInt(2, 3), APInt(1, 1));

  SmallVector<unsigned> mapping = {6, 2};
  APInt expanded =
      circt::detail::expandTruthTableForMergedInputs(tt.table, mapping, 7);
  APInt expected = buildExpectedExpanded(tt.table, mapping, 7);
  EXPECT_EQ(expanded, expected);
}
