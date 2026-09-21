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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class SynthOp<string mnemonic, list<Trait> traits = []> :
    Op<Synth_Dialect, mnemonic, traits>;

class InvertibleOperandsOp<string mnemonic, list<Trait> traits = []>
  : SynthOp<mnemonic,[
      // Share the BooleanLogicOpInterface contract and invertible operand
      // syntax across Synth ops such as `synth.aig.and_inv`.
      SameOperandsAndResultType, Pure,
      DeclareOpInterfaceMethods<BooleanLogicOpInterface, [
        "areInputsPermutationInvariant", "evaluateBooleanLogicWithoutInversion",
        "supportsNumInputs", "createBooleanLogicOp", "getLogicDepthCost",
        "getLogicAreaCost", "emitCNFWithoutInversion"
      ]> ] # traits> {
  // TODO: Restrict to HWIntegerType.
  let arguments = (ins Variadic<AnyType>:$inputs,
                       DenseBoolArrayAttr:$inverted);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    custom<VariadicInvertibleOperands>($inputs, type($result), $inverted, attr-dict)
  }];
}

class ThreeInputOp<string mnemonic, string logicFn, bit symmetric = false,
                   list<Trait> traits = []>
  : InvertibleOperandsOp<mnemonic, [
      PredOpTrait<"requires exactly three operands",
                  CPred<"getOperation()->getNumOperands() == 3">>,
      PredOpTrait<"requires exactly three inversion flags",
                  CPred<"getInverted().size() == 3">>] # traits> {
  let builders = [
    OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c,
                                CArg<"bool", "false">:$invertA,
                                CArg<"bool", "false">:$invertB,
                                CArg<"bool", "false">:$invertC), [{
      SmallVector<bool> inverted{invertA, invertB, invertC};
      return build($_builder, $_state, ValueRange{a, b, c},
                   $_builder.getDenseBoolArrayAttr(inverted));
    }]>
  ];
  let extraClassDefinition = [{
    bool $cppClass::areInputsPermutationInvariant() {
      return }] # !if(symmetric, "true", "false") # [{;
    }
    bool $cppClass::supportsNumInputs(unsigned numInputs) {
      return numInputs == 3;
    }
    int64_t $cppClass::getLogicDepthCost() { return 1; }
    std::optional<uint64_t> $cppClass::getLogicAreaCost() {
      int64_t bitWidth = hw::getBitWidth(getType());
      if (bitWidth < 0)
        return std::nullopt;
      return static_cast<uint64_t>(bitWidth);
    }
    ::llvm::KnownBits $cppClass::computeKnownBits(
        ::llvm::function_ref<const ::llvm::KnownBits &(unsigned)>
            getInputKnownBits) {
      auto a = applyInputInversion(getInputKnownBits(0), isInverted(0));
      auto b = applyInputInversion(getInputKnownBits(1), isInverted(1));
      auto c = applyInputInversion(getInputKnownBits(2), isInverted(2));
      return }] # logicFn # [{(a, b, c);
    }
    ::llvm::APInt $cppClass::evaluateBooleanLogicWithoutInversion(
        ::llvm::ArrayRef<::llvm::APInt> inputs) {
      assert(inputs.size() == 3 && "expected exactly three inputs");
      return }] # logicFn # [{(inputs[0], inputs[1], inputs[2]);
    }
  }];
}

class SymmetricThreeInputOp<string mnemonic, string logicFn,
                            list<Trait> traits = []>
  : ThreeInputOp<mnemonic, logicFn, true, traits>;

def AndInverterOp : InvertibleOperandsOp<"aig.and_inv"> {
  let summary = "AIG dialect AND operation";
  let description = [{
    The `synth.aig.and_inv` operation represents an And-Inverter in the AIG dialect.
    Unlike `comb.and`, operands can be inverted respectively.

    Example:
    ```mlir
      %r1 = synth.aig.and_inv %a, %b: i3
      %r2 = synth.aig.and_inv not %a, %b, not %c : i3
      %r3 = synth.aig.and_inv not %a : i3
    ```

    Traditionally, an And-Node in AIG has two operands. However, `synth.aig.and_inv`
    extends this concept by allowing variadic operands and non-i1 integer types.
    Although the final stage of the synthesis pipeline requires lowering
    everything to i1-binary operands, it's more efficient to progressively lower
    the variadic multibit operations.

    Variadic operands have demonstrated their utility in low-level optimizations
    within the `comb` dialect. Furthermore, in synthesis, it's common practice
    to re-balance the logic path. Variadic operands enable the compiler to
    select more efficient solutions without the need to traverse binary trees
    multiple times.

    The ability to represent multibit operations during synthesis is crucial for
    scalability. This approach enables a form of vectorization, allowing for
    batch processing of logic synthesis when multibit operations are constructed
    in a similar manner.
  }];

  let builders = [OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert),
                            [{
      SmallVector<bool> inverted {invert};
      return build($_builder, $_state, {input}, inverted);
    }]>,
                  OpBuilder<(ins "Value":$lhs, "Value":$rhs,
                                CArg<"bool", "false">:$invertLhs,
                                CArg<"bool", "false">:$invertRhs),
                            [{
      SmallVector<bool> inverted {invertLhs, invertRhs};
      return build($_builder, $_state, {lhs, rhs}, inverted);
    }]>];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let cppNamespace = "::circt::synth::aig";
}

def XorInverterOp : InvertibleOperandsOp<"xor_inv"> {
  let summary = "XOR gate with invertible variadic inputs";
  let description = [{
    The `synth.xor_inv` operation represents a parity node whose operands may be
    individually inverted before the XOR is evaluated.

    Example:
    ```mlir
      %r0 = synth.xor_inv %a, %b : i1
      %r1 = synth.xor_inv not %a, %b, %c : i8
      %r2 = synth.xor_inv not %a : i1
    ```
  }];

  let builders = [
    OpBuilder<(ins "Value":$input, CArg<"bool", "false">:$invert),
                            [{
      SmallVector<bool> inverted {invert};
      return build($_builder, $_state, {input}, inverted);
    }]>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
                                CArg<"bool", "false">:$invertLhs,
                                CArg<"bool", "false">:$invertRhs),
                            [{
      SmallVector<bool> inverted {invertLhs, invertRhs};
      return build($_builder, $_state, {lhs, rhs}, inverted);
    }]>,
    OpBuilder<(ins "ArrayRef<std::pair<Value, bool>>":$operands),
                            [{
      SmallVector<Value> values;
      SmallVector<bool> inverts;
      for (auto &[val, inv] : operands) {
        values.push_back(val);
        inverts.push_back(inv);
      }
      return build($_builder, $_state, values, inverts);
    }]>
    ];
    
    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;

}

def DotOp : ThreeInputOp<"dot", "evaluateDotLogic"> {
  let summary = "Ordered 3-input DOT gate with invertible inputs";
  let description = [{
    The `synth.dot` operation represents the ordered three-input Boolean
    function `x xor (z or (x and y))`, applied after optional per-input
    inversion.

    Unlike common AND, OR, XOR, and majority gates, `synth.dot` is not
    permutation-invariant: its operands play distinct roles in the Boolean
    expression. Dot gates are known to be a promising intermediate
    representation for area optimization, as discussed in "Three-Input Gates
    for Logic Synthesis" (Dewmini Sudara Marakkalage et al., TCAD 2021).

    Example:
    ```mlir
      %r0 = synth.dot %x, %y, %z : i1
      %r1 = synth.dot not %x, %y, not %z : i8
    ```
  }];
}

def ChoiceOp : SynthOp<"choice", [SameOperandsAndResultType, Pure]> {
  let summary = "Synth choice operation";
  let description = [{
    The `synth.choice` operation represents a set of structurally equivalent
    choices, corresponding to the choice nodes introduced in "Reducing
    Structural Bias in Technology Mapping" (ICCAD 2005, S. Chatterjee et al.).

    A compiler may assume all operands are equivalent and replace the operation
    with any one of them at any point in the pipeline.

    Example:
    ```mlir
      %r1 = synth.choice %a, %b : i1
      %r2 = synth.choice %a, %b, %c : i16
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "$inputs attr-dict `:` type($result)";
}

def TimingProbeOp : SynthOp<"timing_probe", [SameOperandsAndResultType, Pure]> {
  let summary = "Assume a required time at a dataflow point";
  let description = [{
    The `synth.timing_probe` operation records a timing assumption at a
    specific dataflow point: the operand value must be stable no later
    than `required_time` picoseconds (max-delay requirement). It is a
    pure pass-through: the result is defined to equal the operand.

    Probes are placed by `synth-derive-probes` (from clock ports and a
    period: every flop D gets `required_time` = period) or by hand, and
    consumed by timing-guided passes (`synth-buffer-high-fanout`
    buffers broadcast drivers feeding violated probes). The buffer pass
    erases all probes when done; nothing downstream of it may observe
    one (mirroring the `create_bundle` lifecycle). No canonicalization
    may forward operands through a probe before its consumer runs.

    Required times are optimistic by construction (input slews, flop
    clock-to-Q and setup times are not modeled); OpenSTA remains ground
    truth, probes are the in-flow action signal.

    Example:
    ```mlir
      %d = synth.timing_probe %driver : i1 {required_time = 10000 : i64}
    ```
  }];
  let arguments = (ins AnyType:$input, I64Attr:$required_time);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$input `:` type($result) attr-dict";
}

def CreateBundleOp : SynthOp<"create_bundle", [Pure]> {
  let summary = "Group values for joint multi-output mapping";
  let description = [{
    The `synth.create_bundle` operation groups values so the technology
    mapper can consider them jointly for multi-output cells (e.g. a sum
    and carry sharing support mapping to one full-adder cell). It is a
    pure pass-through: result `i` is defined to equal operand `i`.
    Extraction creates bundles for shared-support value groups immediately
    pre-mapping; mapping either consumes each bundle into one multi-output
    cell or erases it (falling back to the discrete logic, which stays in
    place). Nothing downstream of mapping may observe one.

    The first two operands are the roots (e.g. sum and carry); any
    further operands record the shared support the pair was extracted
    over. Mapping simulates the joint function over the recorded
    support instead of walking transitively past it, which keeps deep
    cones matchable. Bundles without recorded support simulate
    transitively as before.

    The operation deliberately carries no function data: the joint function
    is derived by simulating the producer cones at match time (ground
    truth), and erasing is a complete fallback. No canonicalization may
    forward operands through it before mapping consumes it.

    Example:
    ```mlir
      %o0, %o1 = synth.create_bundle %sum, %carry : (i1, i1) -> (i1, i1)
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $results)";
}

def MajorityOp : SymmetricThreeInputOp<"majority", "evaluateMajorityLogic"> {
  let summary = "Majority gate with invertible inputs";
  let description = [{
    The `synth.majority` operation computes the majority function of its
    inputs, where each operand can be optionally inverted. 

    The majority function returns 1 when more than half of the inputs are 1,
    and 0 otherwise. For three inputs, it's equivalent to:
    (a & b) | (a & c) | (b & c).


    Example:
  ```mlir
      %r0 = synth.majority %a, %b, %c : i1
      %r1 = synth.majority not %a, %b, not %c : i1
  ```
  }];
}

def OneHotOp : SymmetricThreeInputOp<"onehot", "evaluateOneHotLogic"> {
   let summary = "3-input Onehot gate with invertible inputs";
  let description = [{
    The `synth.onehot` operation computes the one-hot function of its three
    inputs, where each operand can be optionally inverted. The one-hot function
    returns 1 when exactly one of the inputs is 1, and 0 otherwise. For three
    inputs, it is equivalent to:

    `(a & ~b & ~c) | (~a & b & ~c) | (~a & ~b & c)`

    Example:
```mlir
      %r0 = synth.onehot %a, %b, %c : i1
      %r1 = synth.onehot not %a, %b, not %c : i1
```
  }];
}

def MuxInverterOp : ThreeInputOp<"mux_inv", "evaluateMuxLogic"> {
  let summary = "MUX gate with invertible inputs";
  let description = [{
    The `synth.mux_inv` operation represents a 3-input multiplexer node
    whose operands can be individually inverted before evaluation.
    The operation chooses between two values based on a condition:
    given operands (cond, trueValue, falseValue), it calculates `cond ? trueValue : falseValue`,
    which is equivalent to:
    
    `(cond & trueValue) | (~cond & falseValue)`

    Example:
    ```mlir
      %r0 = synth.mux_inv %a, %b, %c : i8
      %r1 = synth.mux_inv not %a, %b, not %c : i8
    ```
  }];
}

def GambleOp : SymmetricThreeInputOp<"gamble", "evaluateGambleLogic"> {
  let summary = "3-input Gamble gate with invertible inputs";
  let description = [{
  The `synth.gamble` operation computes the gamble function of its three
  inputs, where each operand can be optionally inverted. The gamble function
  returns 1 when all inputs are 1 or all inputs are 0, and 0 otherwise.
  For three inputs, it is equivalent to:

  `(a & b & c) | (~a & ~b & ~c)`

  Example:
```mlir
    %r0 = synth.gamble %a, %b, %c : i1
    %r1 = synth.gamble not %a, %b, not %c : i1
```
}];
}

def CutRewritePatternOp : SynthOp<"cut_rewrite_pattern", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "Declarative cut rewrite pattern";

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    MappingCostAttr:$cost
  );

  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def YieldOp : SynthOp<"yield",
                   [Pure, Terminator]> {
  let summary = "Yield synth operations";

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict `:` type($operands)";
}

//===----------------------------------------------------------------------===//
// Liberty sequential state elements
//===----------------------------------------------------------------------===//

def FFOp : SynthOp<"ff", [MemoryEffects<[MemRead, MemWrite]>,
                                             AttrSizedOperandSegments]> {
  let summary = "Liberty edge-triggered flip-flop state element";
  let description = [{
    The `synth.ff` operation is a single-bit edge-triggered state
    element directly modelling a Liberty `ff` group. It carries the state
    that Liberty names `IQ`/`IQN` as its `q`/`qn` results, which always hold
    complementary values.

    All control operands are active-high single-bit values. Any inversion
    required by the Liberty source (for example an active-low `!RESET_B`
    clear) is expressed with `comb` operations outside this op, as are
    scan data muxes. A synchronous enable is an explicit operand (like
    `seq.compreg.ce`); when it is absent the flop always samples.

    - The power-up value is unknown.
    - If `clear` is present and 1, the state becomes 0 regardless of the
      clock. Otherwise, if `preset` is present and 1, the state becomes 1.
      Asserting both at once is undefined.
    - Otherwise the state holds, unless there is no `enable` or `enable`
      is 1, in which case a rising (`edge rise`: 0 to 1) or falling
      (`edge fall`: 1 to 0) edge on `clk` samples `data`.
    - Setup/hold and other timing checks are not modelled here; they live
      in the Liberty timing arcs (`synth.timing.nldm`).

    Example:
    ```mlir
      %q, %qn = synth.ff %next clock %clk edge rise clear %rst : i1, i1
    ```
  }];

  let arguments = (ins I1:$data, I1:$clk, Optional<I1>:$enable,
                       Optional<I1>:$clear, Optional<I1>:$preset,
                       EdgeKindAttr:$edge);
  let results = (outs I1:$q, I1:$qn);
  let assemblyFormat = [{
    $data `clock` $clk `edge` $edge (`enable` $enable^)? (`clear` $clear^)?
    (`preset` $preset^)? attr-dict `:` type($q) `,` type($qn)
  }];
}

def LatchOp : SynthOp<"latch", [MemoryEffects<[MemRead, MemWrite]>,
                                                    AttrSizedOperandSegments]> {
  let summary = "Liberty level-sensitive latch state element";
  let description = [{
    The `synth.latch` operation is a single-bit level-sensitive
    latch directly modelling a Liberty `latch` group, with `q`/`qn` results
    always holding complementary values.

    As with `synth.ff`, all control operands are active-high;
    inversions from the Liberty source are `comb` operations outside this op.

    - The power-up value is unknown.
    - If `clear` is present and 1, the state becomes 0 regardless of
      `enable`. Otherwise, if `preset` is present and 1, the state becomes
      1. Asserting both at once is undefined.
    - Otherwise, while `enable` is 1 the state follows `data`; while it is
      0 the state holds.

    Example:
    ```mlir
      %q, %qn = synth.latch %d enable %en clear %rst : i1, i1
    ```
  }];

  let arguments = (ins I1:$data, I1:$enable, Optional<I1>:$clear,
                       Optional<I1>:$preset);
  let results = (outs I1:$q, I1:$qn);
  let assemblyFormat = [{
    $data `enable` $enable (`clear` $clear^)? (`preset` $preset^)?
    attr-dict `:` type($q) `,` type($qn)
  }];
}

def ClockGateOp : SynthOp<"clock_gate", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Integrated clock-gate cell";
  let description = [{
    The `synth.clock_gate` operation models an integrated clock-gating
    cell: a transparent latch plus an AND gate. It is defined by the
    following decomposition, so `comb` stays `comb` and only the state
    element construct is new (compare `synth.latch`, whose test muxing
    would otherwise have to live inside the op):
    ```mlir
    // test absent: %t = %enable
    %t = comb.or %enable, %test : i1
    // level low: %le = comb.xor %clk, %true; level high: %le = %clk
    %le = <latch enable for transparency while clk == level>
    %l, %ln = synth.latch %t enable %le : i1, i1
    %gclk = comb.and %clk, %l : i1
    ```

    All control operands are active-high single-bit values. The test
    input implements precontrol: it is ORed into the latch data, so the
    gate opens through the latch's normal transparent phase (it is not an
    asynchronous override). The power-up value is unknown. Clock-tree
    timing is not modelled here; it lives in the Liberty timing arcs
    (`synth.timing.nldm`).

    This is the `i1` meeting point for clock gating: unlike
    `seq.clock_gate`, it can live in Liberty library bodies whose pins are
    `i1` wires, and design-side `seq.clock_gate` can normalize to it once
    clocks are treated as values for mapping.

    Example:
    ```mlir
      %gclk = synth.clock_gate %clk enable %ena test %se level low : i1
    ```
  }];

  let arguments = (ins I1:$clk, I1:$enable, Optional<I1>:$testEnable,
                       ClockGateLevelAttr:$level);
  let results = (outs I1:$gclk);
  let assemblyFormat = [{
    $clk `enable` $enable (`test` $testEnable^)? `level` $level attr-dict
    `:` type($gclk)
  }];
}


#endif // CIRCT_DIALECT_SYNTH_SYNTHOPS_TD
