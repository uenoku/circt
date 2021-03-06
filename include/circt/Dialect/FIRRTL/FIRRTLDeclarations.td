//===- FIRRTLDeclarations.td - FIRRTL declaration ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL declarations.
//
//===----------------------------------------------------------------------===//

def InstanceOp : FIRRTLOp<"instance"> {
  let summary = "Instantiate an instance of a module";
  let description = [{
    This represents an instance of a module.  The results are the modules inputs
    and outputs.  The inputs have flip type, the outputs do not.
  }];

  let arguments = (ins FlatSymbolRefAttr:$moduleName,
                       StrArrayAttr:$portNames, OptionalAttr<StrAttr>:$name);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat = "$moduleName attr-dict (`:` type($results)^ )?";

  let verifier = "return ::verifyInstanceOp(*this);";

  let extraClassDeclaration = [{
    /// Lookup the module or extmodule for the symbol.  This returns null on
    /// invalid IR.
    Operation *getReferencedModule();

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo);
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    // Return the result for this instance that corresponds to the specified
    // port name.
    Value getPortNamed(StringRef name);
    Value getPortNamed(StringAttr name);
  }];
}

def CMemOp : FIRRTLOp<"cmem", [/*MemAlloc*/]> {
  let summary = "Define a new cmem";
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def SMemOp : FIRRTLOp<"smem", [/*MemAlloc*/]> {
  let summary = "Define a new smem";
  let arguments = (ins RUWAttr:$ruw, OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = "$ruw attr-dict `:` type($result)";
}

def MemOp : FIRRTLOp<"mem", [/*MemAlloc*/]> {
  let summary = "Define a new mem";
  let arguments = (ins Confined<I32Attr, [IntMinValue<0>]>:$readLatency,
                       Confined<I32Attr, [IntMinValue<1>]>:$writeLatency,
                       Confined<I64Attr, [IntMinValue<1>]>:$depth, RUWAttr:$ruw,
                       StrArrayAttr:$portNames,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs Variadic<FIRRTLType>:$results);

  let assemblyFormat = "$ruw attr-dict `:` type($results)";

  let verifier = "return ::verifyMemOp(*this);";

  let extraClassDeclaration = [{
    enum class PortKind { Read, Write, ReadWrite };

    /// Return the type of a port given the memory depth, type, and kind
    static BundleType getTypeForPort(uint64_t depth, FIRRTLType dataType,
                                     PortKind portKind);

    /// Return the name and kind of ports supported by this memory.
    void getPorts(SmallVectorImpl<std::pair<Identifier, PortKind>> &result);

    /// Return the kind of the specified port or None if the name is invalid.
    Optional<PortKind> getPortKind(StringRef portName);

    /// Return the kind of the specified port number
    Optional<PortKind> getPortKind(size_t resultNo);

    /// Return the data-type field of the memory, the type of each element.
    FIRRTLType getDataType();

    /// Return the port name for the specified result number.
    StringAttr getPortName(size_t resultNo);
    StringRef getPortNameStr(size_t resultNo) {
      return getPortName(resultNo).getValue();
    }

    /// Return the port type for the specified result number.
    FIRRTLType getPortType(size_t resultNo);

    // Return the result for this instance that corresponds to the specified
    // port name.
    Value getPortNamed(StringRef name) {
      return getPortNamed(StringAttr::get(getContext(), name));
    }
    Value getPortNamed(StringAttr name);
  }];
}

def NodeOp : FIRRTLOp<"node", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "No-op to name a value";
  let description = [{
    A node is simply a named intermediate value in a circuit. The node must
    be initialized to a value with a passive type and cannot be connected to.
    Nodes are often used to split a complicated compound expression into named
    subexpressions.

    ```
      %result = firrtl.node %input : t1
    ```
    }];

  let arguments = (ins PassiveType:$input, OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat = "$input custom<ImplicitSSAName>(attr-dict) `:` type($input)";
}

def RegOp : FIRRTLOp<"reg", [/*MemAlloc*/]> {
  let summary = "Define a new register";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.reg %clockVal : t1
    ```
    }];

  let arguments = (ins ClockType:$clockVal, OptionalAttr<StrAttr>:$name);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, $result)";
}

def RegResetOp : FIRRTLOp<"regreset", [/*MemAlloc*/]> {
  let summary = "Define a new register with a reset";
  let description = [{
    Declare a new register:
    ```
      %name = firrtl.regreset %clockVal, %resetSignal, %resetValue : t1
    ```
    }];

  let arguments = (ins ClockType:$clockVal, ResetType:$resetSignal,
                       PassiveType:$resetValue, OptionalAttr<StrAttr>:$name);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
     "operands attr-dict `:` functional-type(operands, $result)";
}

def WireOp : FIRRTLOp<"wire", []> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrtl.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs FIRRTLType:$result);

  let builders = [
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringRef", "{}">:$name), [{
      return build($_builder, $_state, elementType,
                   $_builder.getStringAttr(name));
    }]>
  ];

  let assemblyFormat = "custom<ImplicitSSAName>(attr-dict) `:` type($result)";
}
