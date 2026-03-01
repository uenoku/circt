//===- SynthAttributes.td - Attributes for Synth dialect --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
#define CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"

def NLDMTimeUnitAttr : AttrDef<Synth_Dialect, "NLDMTimeUnit"> {
  let mnemonic = "nldm_time_unit";
  let summary = "NLDM time unit scale in picoseconds";
  let parameters = (ins "mlir::FloatAttr":$picoseconds);
  let assemblyFormat = "`<` $picoseconds `>`";
}

def NLDMArcAttr : AttrDef<Synth_Dialect, "NLDMArc"> {
  let mnemonic = "nldm_arc";
  let summary = "Normalized NLDM arc metadata";
  let parameters = (ins
    "mlir::StringAttr":$relatedPin,
    "mlir::StringAttr":$toPin,
    "mlir::StringAttr":$timingSense,
    "mlir::ArrayAttr":$cellRiseValues,
    "mlir::ArrayAttr":$cellFallValues
  );
  let assemblyFormat = [{
    `<` $relatedPin `,` $toPin `,` $timingSense `,` $cellRiseValues `,`
    $cellFallValues `>`
  }];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
