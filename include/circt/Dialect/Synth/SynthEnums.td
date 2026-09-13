//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Synth dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD
#define CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD

include "mlir/IR/EnumAttr.td"

let cppNamespace = "circt::synth" in {

def Polarity_Positive : I32EnumAttrCase<"Positive", 0, "positive">;
def Polarity_Negative : I32EnumAttrCase<"Negative", 1, "negative">;

def PolarityKindAttr : I32EnumAttr<"PolarityKind", "timing arc polarity",
  [Polarity_Positive, Polarity_Negative]>;

def Edge_Rise : I32EnumAttrCase<"Rise", 0, "rise">;
def Edge_Fall : I32EnumAttrCase<"Fall", 1, "fall">;

def EdgeKindAttr : I32EnumAttr<"EdgeKind",
  "Liberty sequential clock edge", [Edge_Rise, Edge_Fall]>;

def ClockGateLevel_Low : I32EnumAttrCase<"Low", 0, "low">;
def ClockGateLevel_High : I32EnumAttrCase<"High", 1, "high">;

def ClockGateLevelAttr : I32EnumAttr<"ClockGateLevel",
  "Clock-gate latch transparency level", [ClockGateLevel_Low,
                                             ClockGateLevel_High]>;

}

#endif // CIRCT_DIALECT_SYNTH_SYNTHENUMS_TD
