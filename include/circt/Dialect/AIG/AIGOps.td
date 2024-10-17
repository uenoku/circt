//===- AIGOps.td - AIG Op Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIG Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_OPS_TD
#define CIRCT_DIALECT_AIG_OPS_TD

include "circt/Dialect/AIG/AIG.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class AIGOp<string mnemonic, list<Trait> traits = []> :
  Op<AIG_Dialect, mnemonic, traits>;

def AndInverterOp : AIGOp<"and_inv", [SameOperandsAndResultType]> {
  let summary = "AIG dialect AND operation";
  let description = [{
    The `aig.and_inv` operation represents a logical AND in the AIG dialect.
    Unlike comb.and, operands can be inverted.

    Example:
    ```mlir
      %result = aig.and_inv_inv %a, %b : i3
      %result = aig.and_inv_inv not %a, %b : i3
      %result = aig.and_inv_inv %a, not %b : i3
      %result = aig.and_inv_inv not %a, not %b : i3
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs, DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, CArg<"bool", "false">:$invertLhs,
                   CArg<"bool", "false">:$invertRhs), [{
      ArrayRef<bool> inverted = {invertLhs, invertRhs};
      return build($_builder, $_state, {lhs, rhs}, inverted);
    }]>
  ];

  let extraClassDeclaration = [{
    APInt evaluate(ArrayRef<APInt> inputs);
    bool isInverted(size_t idx) {
      return getInverted()[idx];
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def CutOp : AIGOp<"cut", [IsolatedFromAbove, SingleBlock]> {
  let summary = "AIG dialect Cut operation";
  let description = [{
    The `aig.cut` operation represents a cut in the And-Inverter-Graph.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $results) $body
  }];
}

def OutputOp : AIGOp<"output", [Terminator,
                                ReturnLike, ParentOneOf<["CutOp"]>]> {
  let summary = "AIG dialect Output operation";
  let description = [{
    The `aig.output` operation represents out edges of a cut.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

#endif // CIRCT_DIALECT_AIG_OPS_TD
