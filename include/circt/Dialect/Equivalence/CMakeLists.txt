##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## TableGen for the Equivalence dialect (moved from Tamagoyaki).
## Keeps Tamagoyaki's file layout (Ops/Attrs/Passes split) so the moved
## sources stay verbatim apart from include paths.
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Equivalence equivalence)
add_circt_dialect_doc(Equivalence equivalence)

# Tamagoyaki-style extra outputs (Ops/Attrs) expected by the moved headers.
set(LLVM_TARGET_DEFINITIONS EquivalenceDialect.td)
mlir_tablegen(EquivalenceOps.h.inc -gen-op-decls)
mlir_tablegen(EquivalenceOps.cpp.inc -gen-op-defs)
mlir_tablegen(EquivalenceAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=equivalence)
mlir_tablegen(EquivalenceAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=equivalence)
mlir_tablegen(EquivalencePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTEquivalenceExtraIncGen)

add_circt_doc(EquivalencePasses EquivalencePasses -gen-pass-doc)
