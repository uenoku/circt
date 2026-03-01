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
    "mlir::ArrayAttr":$cellRiseIndex1,
    "mlir::ArrayAttr":$cellRiseIndex2,
    "mlir::ArrayAttr":$cellRiseValues,
    "mlir::ArrayAttr":$cellFallIndex1,
    "mlir::ArrayAttr":$cellFallIndex2,
    "mlir::ArrayAttr":$cellFallValues,
    "mlir::ArrayAttr":$riseTransitionIndex1,
    "mlir::ArrayAttr":$riseTransitionIndex2,
    "mlir::ArrayAttr":$riseTransitionValues,
    "mlir::ArrayAttr":$fallTransitionIndex1,
    "mlir::ArrayAttr":$fallTransitionIndex2,
    "mlir::ArrayAttr":$fallTransitionValues
  );
  let assemblyFormat = [{
    `<` $relatedPin `,` $toPin `,` $timingSense `,` $cellRiseIndex1 `,`
    $cellRiseIndex2 `,` $cellRiseValues `,` $cellFallIndex1 `,`
    $cellFallIndex2 `,` $cellFallValues `,` $riseTransitionIndex1 `,`
    $riseTransitionIndex2 `,` $riseTransitionValues `,`
    $fallTransitionIndex1 `,` $fallTransitionIndex2 `,`
    $fallTransitionValues `>`
  }];
}

def CCSPilotArcAttr : AttrDef<Synth_Dialect, "CCSPilotArc"> {
  let mnemonic = "ccs_pilot_arc";
  let summary = "Normalized CCS pilot arc waveform metadata";
  let parameters = (ins
    "mlir::StringAttr":$relatedPin,
    "mlir::StringAttr":$toPin,
    "mlir::ArrayAttr":$currentRiseTimes,
    "mlir::ArrayAttr":$currentRiseValues,
    "mlir::ArrayAttr":$currentFallTimes,
    "mlir::ArrayAttr":$currentFallValues
  );
  let assemblyFormat = [{
    `<` $relatedPin `,` $toPin `,` $currentRiseTimes `,` $currentRiseValues
    `,` $currentFallTimes `,` $currentFallValues `>`
  }];
}

def CCSPilotReceiverAttr : AttrDef<Synth_Dialect, "CCSPilotReceiver"> {
  let mnemonic = "ccs_pilot_receiver";
  let summary = "Normalized CCS pilot receiver capacitance metadata";
  let parameters = (ins
    "mlir::StringAttr":$relatedPin,
    "mlir::StringAttr":$toPin,
    "mlir::ArrayAttr":$cap1RiseIndex1,
    "mlir::ArrayAttr":$cap1RiseIndex2,
    "mlir::ArrayAttr":$cap1RiseValues,
    "mlir::ArrayAttr":$cap2RiseIndex1,
    "mlir::ArrayAttr":$cap2RiseIndex2,
    "mlir::ArrayAttr":$cap2RiseValues,
    "mlir::ArrayAttr":$cap1FallIndex1,
    "mlir::ArrayAttr":$cap1FallIndex2,
    "mlir::ArrayAttr":$cap1FallValues,
    "mlir::ArrayAttr":$cap2FallIndex1,
    "mlir::ArrayAttr":$cap2FallIndex2,
    "mlir::ArrayAttr":$cap2FallValues
  );
  let assemblyFormat = [{
    `<` $relatedPin `,` $toPin
    `,` $cap1RiseIndex1 `,` $cap1RiseIndex2 `,` $cap1RiseValues
    `,` $cap2RiseIndex1 `,` $cap2RiseIndex2 `,` $cap2RiseValues
    `,` $cap1FallIndex1 `,` $cap1FallIndex2 `,` $cap1FallValues
    `,` $cap2FallIndex1 `,` $cap2FallIndex2 `,` $cap2FallValues `>`
  }];
}

#endif // CIRCT_DIALECT_SYNTH_SYNTHATTRIBUTES_TD
