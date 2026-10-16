//===- DropConst.cpp - Check and remove const types -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropConst pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_DROPCONST
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

/// Returns null type if no conversion is needed.
static FIRRTLBaseType convertType(FIRRTLBaseType type) {
  auto nonConstType = type.getAllConstDroppedType();
  return nonConstType != type ? nonConstType : FIRRTLBaseType{};
}

/// Returns null type if no conversion is needed.
static Type convertType(Type type) {
  if (auto base = type_dyn_cast<FIRRTLBaseType>(type)) {
    return convertType(base);
  }

  if (auto refType = type_dyn_cast<RefType>(type)) {
    if (auto converted = convertType(refType.getType()))
      return RefType::get(converted, refType.getForceable(),
                          refType.getLayer());
  }

  return {};
}

namespace {
class DropConstPass : public circt::firrtl::impl::DropConstBase<DropConstPass> {
  void runOnOperation() override {

    // Rewrite module signatures and bodies in parallel. Each module has its own
    // cache so that repeated aggregate types are only converted once, without
    // locking.
    mlir::parallelForEach(
        &getContext(), getOperation().getOps<FModuleLike>(), [](auto mod) {
          DenseMap<Type, Type> typeCache;
          auto convert = [&](Type type) -> Type {
            // Integer types are cheap to convert and may have many distinct
            // widths. Keep them out of the cache.
            if (auto intType = dyn_cast<IntType>(type))
              return intType.isConst() ? intType.getConstType(false) : Type{};

            auto [it, inserted] = typeCache.try_emplace(type);
            if (inserted)
              it->second = convertType(type);
            return it->second;
          };

          // Update the module signature with non-'const' ports.
          SmallVector<Attribute> portTypes;
          portTypes.reserve(mod.getNumPorts());
          bool convertedAny = false;
          llvm::transform(mod.getPortTypes(), std::back_inserter(portTypes),
                          [&](Attribute type) -> Attribute {
                            if (auto convertedType =
                                    convert(cast<TypeAttr>(type).getValue())) {
                              convertedAny = true;
                              return TypeAttr::get(convertedType);
                            }
                            return type;
                          });
          if (convertedAny)
            mod.setPortTypesAttr(ArrayAttr::get(mod.getContext(), portTypes));

          // Convert the module body if present. Check for FModuleOp
          // specifically as there's no "hasBody()".
          if (!isa<FModuleOp>(mod))
            return;
          mod->walk([&](Operation *op) {
            if (auto constCastOp = dyn_cast<ConstCastOp>(op)) {
              // Remove any `ConstCastOp`, replacing results with inputs
              constCastOp.getResult().replaceAllUsesWith(
                  constCastOp.getInput());
              constCastOp->erase();
              return;
            }

            // Convert any block arguments
            for (auto &region : op->getRegions())
              for (auto &block : region.getBlocks())
                for (auto argument : block.getArguments())
                  if (auto convertedType = convert(argument.getType()))
                    argument.setType(convertedType);

            for (auto result : op->getResults())
              if (auto convertedType = convert(result.getType()))
                result.setType(convertedType);
          });
        });

    markAnalysesPreserved<InstanceGraph>();
  }
};
} // namespace
