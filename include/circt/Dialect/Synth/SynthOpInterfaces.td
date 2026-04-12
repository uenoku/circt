//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD
#define CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def BooleanLogicOpInterface : OpInterface<"BooleanLogicOpInterface"> {
  let cppNamespace = "::circt::synth";

  let description = [{
    Common interface for bitwise boolean-logic operations.
  }];

  let methods = [
    InterfaceMethod<[{
        Get the number of logical inputs to this operation.
      }],
      "unsigned", "getNumLogicInputs", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp->getNumOperands();
      }]
    >,
    InterfaceMethod<[{
        Get the logical input at the specified index and whether it is
        inverted before the operation is evaluated.
      }],
      "::std::pair<::mlir::Value, bool>", "getInput", (ins "unsigned":$index), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return {concreteOp->getOperand(index), concreteOp.isInverted(index)};
      }]
    >,
    InterfaceMethod<[{
        Get the logical input value at the specified index.
      }],
      "::mlir::Value", "getInputValue", (ins "unsigned":$index), "",
      [{
        return $_op.getInput(index).first;
      }]
    >,
    InterfaceMethod<[{
        Return whether the specified logical input is inverted before the
        operation is evaluated.
      }],
      "bool", "isInputInverted", (ins "unsigned":$index), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.isInverted(index);
      }]
    >,
    InterfaceMethod<[{
        Get the inversion flags for all logical inputs.
      }],
      "::llvm::ArrayRef<bool>", "getInputInversions", (ins), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.getInverted();
      }]
    >,
    InterfaceMethod<[{
        Return whether the operation semantics are invariant under
        permutation of logical input pairs. Reordering a logical input means
        moving both the input value and its inversion flag together.
      }],
      "bool", "areInputsPermutationInvariant", (ins), "",
      [{
        return true;
      }]
    >,
    InterfaceMethod<[{
        Evaluate the boolean logic operation for the provided raw operand
        inputs. The callback returns the raw operand value for input `i`;
        implementations are responsible for applying any per-input inversion
        described by the operation itself.
      }],
      "::llvm::APInt", "evaluateBooleanLogic",
      (ins "::llvm::function_ref<const ::llvm::APInt &(unsigned)>":$getInputValue), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.evaluate(getInputValue);
      }]
    >,
    InterfaceMethod<[{
        Compute known bits for the result from the logical inputs.
        The callback returns known bits for the raw input value `i`;
        implementations are responsible for applying any per-input inversion
        described by the operation itself.
      }],
      "::llvm::KnownBits", "computeKnownBits",
      (ins "::llvm::function_ref<const ::llvm::KnownBits &(unsigned)>":$getInputKnownBits), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        return concreteOp.computeKnownBits(getInputKnownBits);
      }]
    >,
    InterfaceMethod<[{
        Clone this operation with the provided inputs while preserving the
        existing inversion flags.
      }],
      "::mlir::Operation *", "cloneWithSameInversion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::ValueRange":$newInputs), "",
      [{
        auto concreteOp = mlir::cast<ConcreteOp>($_op);
        auto resultType =
            newInputs.empty() ? concreteOp.getType() : newInputs.front().getType();
        return ConcreteOp::create(builder, concreteOp.getLoc(),
                                  resultType, newInputs,
                                  concreteOp.getInverted())
            .getOperation();
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHOPINTERFACES_TD
