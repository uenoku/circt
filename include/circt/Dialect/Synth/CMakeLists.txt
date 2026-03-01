##===----------------------------------------------------------------------===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Synth synth)
add_circt_dialect_doc(Synth synth)

set(LLVM_TARGET_DEFINITIONS Synth.td)
mlir_tablegen(SynthAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=synth)
mlir_tablegen(SynthAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=synth)
add_public_tablegen_target(CIRCTSynthAttributesIncGen)
add_dependencies(circt-headers CIRCTSynthAttributesIncGen)

add_subdirectory(Transforms)
