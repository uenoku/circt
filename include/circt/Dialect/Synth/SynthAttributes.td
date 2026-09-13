//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
#define CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD


include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "circt/Dialect/Synth/SynthEnums.td"
include "circt/Dialect/Synth/SynthDialect.td"

// A single resolved 2D NLDM lookup table. Values are stored in row-major order
// with dimensions index1.size x index2.size.
def NLDMTableAttr : AttrDef<Synth_Dialect, "NLDMTable"> {
  let mnemonic = "nldm_table";
  let summary = "Resolved 2D NLDM lookup table";
  let parameters = (ins
    "::mlir::DenseF64ArrayAttr":$index1,
    "::mlir::DenseF64ArrayAttr":$index2,
    "::mlir::DenseF64ArrayAttr":$values
  );

  let assemblyFormat = "`<` $index1 `,` $index2 `,` $values `>`";
}

def NLDMTimingArcAttr : AttrDef<Synth_Dialect, "NLDMTimingArc"> {
  let mnemonic = "nldm_timing_arc";
  let summary = "NLDM timing arc with delay and transition lookup tables";
  let parameters = (ins
    "::mlir::StringAttr":$pin,
    "::mlir::StringAttr":$relatedPin,
    "::mlir::StringAttr":$timingSense,
    "NLDMTableAttr":$cellRise,
    "NLDMTableAttr":$cellFall,
    "NLDMTableAttr":$riseTransition,
    "NLDMTableAttr":$fallTransition
  );
  let assemblyFormat =
    "`<` $pin `,` $relatedPin `,` $timingSense `,` "
    "$cellRise `,` $cellFall `,` "
    "$riseTransition `,` $fallTransition `>`";
}

def PolarityAttr : AttrDef<Synth_Dialect, "Polarity"> {
  let mnemonic = "polarity";
  let summary = "Timing arc polarity";
  let parameters = (ins
    "PolarityKind":$value
  );
  let assemblyFormat = "`<` $value `>`";
}

def LinearTimingArcAttr : AttrDef<Synth_Dialect, "LinearTimingArc"> {
  let mnemonic = "linear_timing_arc";
  let summary = "A single simplified linear timing arc";
  let parameters = (ins
    "int64_t":$intrinsic,
    "int64_t":$sensitivity,
    "PolarityAttr":$polarity
  );
  let assemblyFormat =
    "`<` $intrinsic `,` $sensitivity `,` $polarity `>`";
}

def MappingCostAttr : AttrDef<Synth_Dialect, "MappingCost"> {
  let mnemonic = "mapping_cost";
  let summary = "Simplified timing and area cost for tech mapping";
  let parameters = (ins
    "::mlir::FloatAttr":$area,
    "::mlir::ArrayAttr":$arcs,
    OptionalParameter<"::mlir::DictionaryAttr">:$inputCaps
  );
  let genVerifyDecl = 1;
  let assemblyFormat =
    "`<` `area` `=` $area "
    "`,` `arcs` `=` $arcs "
    "(`,` `input_caps` `=` $inputCaps^)? `>`";
}

// A single SRAM macro port with dedicated read or write behavior. Pin names
// refer to ports of the extern module; an empty enable or mask means the
// port has none. Widths are in bits; latency is in cycles.
def SRAMPortAttr : AttrDef<Synth_Dialect, "SRAMPort"> {
  let mnemonic = "sram_port";
  let summary = "A single SRAM macro port";
  let parameters = (ins
    "SRAMPortKind":$kind,
    "::mlir::StringAttr":$clock,
    "::mlir::StringAttr":$enable,
    "::mlir::StringAttr":$address, "int64_t":$addrWidth,
    "::mlir::StringAttr":$data, "int64_t":$dataWidth,
    "::mlir::StringAttr":$mask, "int64_t":$maskWidth,
    "int64_t":$latency
  );
  let assemblyFormat =
    "`<` $kind `,` $clock `,` $enable `,` $address `,` $addrWidth `,` "
    "$data `,` $dataWidth `,` $mask `,` $maskWidth `,` $latency `>`";
}

// An SRAM macro descriptor for memory mapping: geometry plus port list.
def SRAMMacroAttr : AttrDef<Synth_Dialect, "SRAMMacro"> {
  let mnemonic = "sram_macro";
  let summary = "SRAM macro descriptor for memory mapping";
  let parameters = (ins
    "int64_t":$width,
    "int64_t":$depth,
    "::mlir::ArrayAttr":$ports
  );
  let assemblyFormat = "`<` $width `x` $depth `,` $ports `>`";
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
