//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the Synth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
#define CIRCT_DIALECT_SYNTH_SYNTHOPS_TD

include "circt/Dialect/Synth/Synth.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class SynthOp<string mnemonic, list<Trait> traits = []> :
    Op<Synth_Dialect, mnemonic, traits>;

def MajorityInverterOp : SynthOp<"maj_inv", [SameOperandsAndResultType, Pure]> {
  let summary = "Synth dialect Majority-Inverter operation";
  let description = [{
    The `synth.maj_inv` operation represents a Majority-Inverter in the Synth dialect.
    This operation computes the majority function of its inputs, where operands can be 
    inverted respectively.

    The majority function returns 1 when more than half of the inputs are 1, and 0 otherwise.
    For three inputs, it's equivalent to: (a & b) | (a & c) | (b & c).

    Example:
    ```mlir
      %r1 = synth.maj_inv %a, %b, %c : i1
      %r2 = synth.maj_inv not %a, %b, not %c : i1
      %r3 = synth.maj_inv %a, %b, %c, %d, %e : i3
    ```

    Similar to `aig.and_inv`, this operation extends the traditional binary/ternary
    majority gate concept by allowing variadic operands and non-i1 integer types.
    This enables more efficient synthesis transformations and optimizations.

    The operation is particularly useful in synthesis as majority gates are fundamental
    building blocks in many logic synthesis algorithms and can lead to more efficient
    implementations compared to AND/OR compositions.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);

  // NOTE: Custom assembly format is needed to pretty print the `inverted`
  // attribute.
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert), [{
      SmallVector<bool> inverted {invert};
      return build($_builder, $_state, {input}, inverted);
    }]>,
    OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c, 
                   CArg<"bool", "false">:$invertA,
                   CArg<"bool", "false">:$invertB, 
                   CArg<"bool", "false">:$invertC), [{
      SmallVector<bool> inverted {invertA, invertB, invertC};
      return build($_builder, $_state, {a, b, c}, inverted);
    }]>
  ];

  let extraClassDeclaration = [{
    // Evaluate the operation with the given input values.
    APInt evaluate(ArrayRef<APInt> inputs);

    // Check if the input is inverted.
    bool isInverted(size_t idx) {
      return getInverted()[idx];
    }
  }];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
