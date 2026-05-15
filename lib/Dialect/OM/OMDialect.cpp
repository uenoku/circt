//===- OMDialect.cpp - Object Model dialect definition --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect definition.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMOps.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Builders.h"

#include "circt/Dialect/OM/OMDialect.cpp.inc"

namespace {
enum OMAttributeCode {
  kPathAttr = 0,
  kIntegerAttr = 1,
};

struct OMBytecodeDialectInterface : public mlir::BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;

  mlir::Attribute
  readAttribute(mlir::DialectBytecodeReader &reader) const final {
    uint64_t code;
    if (mlir::failed(reader.readVarInt(code)))
      return {};

    auto *context = getContext();
    switch (code) {
    case kPathAttr: {
      llvm::SmallVector<circt::om::PathElement> path;
      uint64_t size;
      if (mlir::failed(reader.readVarInt(size)))
        return {};
      path.reserve(size);
      for (uint64_t i = 0; i != size; ++i) {
        mlir::StringAttr module;
        mlir::StringAttr instance;
        if (mlir::failed(reader.readAttribute(module)) ||
            mlir::failed(reader.readAttribute(instance)))
          return {};
        path.emplace_back(module, instance);
      }
      return circt::om::PathAttr::get(context, path);
    }
    case kIntegerAttr: {
      mlir::IntegerAttr value;
      if (mlir::failed(reader.readAttribute(value)))
        return {};
      return circt::om::IntegerAttr::get(context, value);
    }
    default:
      reader.emitError() << "unknown om attribute code: " << code;
      return {};
    }
  }

  mlir::LogicalResult
  writeAttribute(mlir::Attribute attr,
                 mlir::DialectBytecodeWriter &writer) const final {
    if (auto path = mlir::dyn_cast<circt::om::PathAttr>(attr)) {
      writer.writeVarInt(kPathAttr);
      writer.writeList(path.getPath(),
                       [&](const circt::om::PathElement &element) {
                         writer.writeAttribute(element.module);
                         writer.writeAttribute(element.instance);
                       });
      return mlir::success();
    }

    if (auto integer = mlir::dyn_cast<circt::om::IntegerAttr>(attr)) {
      writer.writeVarInt(kIntegerAttr);
      writer.writeAttribute(integer.getValue());
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void circt::om::OMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/OM/OM.cpp.inc"
      >();

  registerTypes();
  registerAttributes();
  addInterfaces<OMBytecodeDialectInterface>();
}

mlir::Operation *
circt::om::OMDialect::materializeConstant(mlir::OpBuilder &builder,
                                          mlir::Attribute value,
                                          mlir::Type type, mlir::Location loc) {
  if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(value))
    if (typedAttr.getType() == type)
      return ConstantOp::create(builder, loc, typedAttr);
  return nullptr;
}

// Provide implementations for the enums we use.
#include "circt/Dialect/OM/OMEnums.cpp.inc"
