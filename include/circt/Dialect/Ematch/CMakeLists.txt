##===- CMakeLists.txt - ---------------------------------------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
## TableGen for the Ematch dialect (moved from Tamagoyaki).
##
##===----------------------------------------------------------------------===//

add_circt_dialect(Ematch ematch)
add_circt_dialect_doc(Ematch ematch)

set(LLVM_TARGET_DEFINITIONS EmatchDialect.td)
mlir_tablegen(EmatchOps.h.inc -gen-op-decls)
mlir_tablegen(EmatchOps.cpp.inc -gen-op-defs)
mlir_tablegen(EmatchPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTEmatchExtraIncGen)

add_circt_doc(EmatchPasses EmatchPasses -gen-pass-doc)
