//===- LegalizeNames.cpp - Name Legalization for ExportVerilog ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This renames modules and variables to avoid conflicts with keywords and other
// declarations.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sv;
using namespace hw;
using namespace ExportVerilog;

StringAttr ExportVerilog::getDeclarationName(Operation *op) {
  if (auto attr = op->getAttrOfType<StringAttr>("name"))
    return attr;
  if (auto attr = op->getAttrOfType<StringAttr>("instanceName"))
    return attr;
  if (auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return attr;
  return {};
}

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

/// Given a name that may have collisions or invalid symbols, return a
/// replacement name to use, or null if the original name was ok.
StringRef NameCollisionResolver::getLegalName(StringRef originalName) {
  return legalizeName(originalName, usedNames, nextGeneratedNameID);
}

//===----------------------------------------------------------------------===//
// FieldNameResolver
//===----------------------------------------------------------------------===//

/// This constructs the legalized type for given `type` with cache.
Type FieldNameResolver::getLegalizedType(Type type) {
  auto it = legalizedTypes.find(type);
  if (it != legalizedTypes.end())
    return it->second;
  Type legalizedType =
      TypeSwitch<Type, Type>(type)
          .Case<StructType>([&](auto fieldType) -> Type {
            auto elements = fieldType.getElements();
            bool changed = false;
            SmallVector<hw::detail::FieldInfo> newFields;
            newFields.reserve(elements.size());
            for (auto fieldInfo : elements) {
              auto newFieldInfo = getRenamedFieldInfo(fieldInfo);
              changed |= newFieldInfo != fieldInfo;
              newFields.push_back(newFieldInfo);
            }

            return changed ? decltype(fieldType)::get(fieldType.getContext(),
                                                      newFields)
                           : fieldType;
          })
          // Other than struct types, just recursively apply to
          // their subtypes.
          .Case<ArrayType, UnpackedArrayType>([&](auto parentType) -> Type {
            auto elem = getLegalizedType(parentType.getElementType());
            if (elem == parentType.getElementType())
              return type;

            return decltype(parentType)::get(elem, parentType.getSize());
          })
          .Case<InOutType>([&](auto inoutType) -> Type {
            auto elem = getLegalizedType(inoutType.getElementType());
            if (elem == inoutType.getElementType())
              return type;
            return InOutType::get(elem);
          })
          .Case<FunctionType>([&](FunctionType functionType) -> Type {
            auto inputs = functionType.getInputs();
            auto results = functionType.getResults();
            bool changed = false;
            auto renameTypes = [&](auto types) {
              SmallVector<Type> newInputs;
              newInputs.reserve(types.size());
              llvm::transform(types, std::back_inserter(newInputs),
                              [&](Type type) {
                                auto newType = getLegalizedType(type);
                                changed |= newType != type;
                                return newType;
                              });
              return newInputs;
            };
            auto newInputs = renameTypes(inputs);
            auto newResults = renameTypes(results);
            return changed ? FunctionType::get(functionType.getContext(),
                                               newInputs, newResults)
                           : type;
          })
          .Case<TypeAliasType>([&](TypeAliasType aliasType) -> Type {
            auto innerType = getLegalizedType(aliasType.getInnerType());
            if (innerType == aliasType.getInnerType())
              return type;
            return TypeAliasType::get(aliasType.getRef(), innerType);
          })
          .Default([](auto baseType) { return baseType; });
  setLegalizedType(type, legalizedType);
  return legalizedType;
}

void FieldNameResolver::setRenamedFieldName(StringAttr fieldName,
                                            StringAttr newFieldName) {
  renamedFieldNames[fieldName] = newFieldName;
  usedFieldNames.insert(newFieldName);
}

void FieldNameResolver::setLegalizedType(Type type, Type newType) {
  legalizedTypes[type] = newType;
}

StringAttr FieldNameResolver::getRenamedFieldName(StringAttr fieldName) {
  auto it = renamedFieldNames.find(fieldName);
  if (it != renamedFieldNames.end())
    return it->second;

  // If a field name is not verilog name or used already, we have to rename it.
  bool hasToBeRenamed = !sv::isNameValid(fieldName.getValue()) ||
                        usedFieldNames.count(fieldName.getValue());

  if (!hasToBeRenamed) {
    setRenamedFieldName(fieldName, fieldName);
    return fieldName;
  }

  StringRef newFieldName = sv::legalizeName(
      fieldName.getValue(), usedFieldNames, nextGeneratedNameID);

  auto newFieldNameAttr = StringAttr::get(fieldName.getContext(), newFieldName);

  setRenamedFieldName(fieldName, newFieldNameAttr);
  return newFieldNameAttr;
}

hw::detail::FieldInfo
FieldNameResolver::getRenamedFieldInfo(hw::detail::FieldInfo fieldInfo) {
  fieldInfo.name = getRenamedFieldName(fieldInfo.name);
  fieldInfo.type = getLegalizedType(fieldInfo.type);
  return fieldInfo;
}

void FieldNameResolver::legalizeToplevelOperation(Operation *op) {
  // Legalize operations including `op` itself.
  op->walk([&](Operation *op) { legalizeOperationTypes(op); });
}

void FieldNameResolver::legalizeOperationTypes(Operation *op) {
  // Legalize result types.
  for (auto result : op->getResults()) {
    auto newType = getLegalizedType(result.getType());
    if (result.getType() != newType)
      result.setType(newType);
  }

  // Rename field names referred in operations.
  if (isa<sv::StructFieldInOutOp, hw::StructExtractOp, hw::StructInjectOp>(
          op)) {
    auto fieldNameAttr = op->getAttr("field").cast<StringAttr>();
    auto newFieldNameAttr = getRenamedFieldName(fieldNameAttr);
    if (fieldNameAttr != newFieldNameAttr)
      op->setAttr("field", newFieldNameAttr);
    return;
  }

  if (auto module = dyn_cast<HWModuleOp>(op)) {
    // Legalize module types.
    auto type = getLegalizedType(module.getType());
    if (type != module.getType())
      module.setType(type.cast<mlir::FunctionType>());

    // We have to replace argument types as well.
    for (auto arg : module.getBody().getArguments()) {
      auto newType = getLegalizedType(arg.getType());
      if (arg.getType() != newType)
        arg.setType(newType);
    }
    return;
  }

  // Legalize external module types.
  if (auto extmodule = dyn_cast<HWModuleExternOp>(op)) {
    auto type = getLegalizedType(extmodule.getType());
    if (type != extmodule.getType())
      extmodule.setType(type.cast<mlir::FunctionType>());
    return;
  }

  // Legalize interface signal types.
  if (auto interfaceSignal = dyn_cast<InterfaceSignalOp>(op)) {
    auto type = getLegalizedType(interfaceSignal.type());
    if (type != interfaceSignal.type())
      interfaceSignal.typeAttr(TypeAttr::get(type));
    return;
  }
}

//===----------------------------------------------------------------------===//
// GlobalNameResolver
//===----------------------------------------------------------------------===//

namespace circt {
namespace ExportVerilog {
/// This class keeps track of modules and interfaces that need to be renamed, as
/// well as module ports and parameters that need to be renamed.  This can
/// happen either due to conflicts between them or due to a conflict with a
/// Verilog keyword.
///
/// Once constructed, this is immutable.
class GlobalNameResolver {
public:
  /// Construct a GlobalNameResolver and do the initial scan to populate and
  /// unique the module/interfaces and port/parameter names.
  GlobalNameResolver(mlir::ModuleOp topLevel);

  GlobalNameTable takeGlobalNameTable() { return std::move(globalNameTable); }

private:
  /// Check to see if the port names of the specified module conflict with
  /// keywords or themselves.  If so, add the replacement names to
  /// globalNameTable.
  void legalizeModuleNames(HWModuleOp module);
  void legalizeInterfaceNames(InterfaceOp interface);

  /// Set of globally visible names, to ensure uniqueness.
  NameCollisionResolver globalNameResolver;

  /// This keeps track of globally visible names like module parameters.
  GlobalNameTable globalNameTable;

  /// This keeps track of field names of struct types.
  FieldNameResolver fieldNameResolver;

  GlobalNameResolver(const GlobalNameResolver &) = delete;
  void operator=(const GlobalNameResolver &) = delete;
};
} // namespace ExportVerilog
} // namespace circt

/// Construct a GlobalNameResolver and do the initial scan to populate and
/// unique the module/interfaces and port/parameter names.
GlobalNameResolver::GlobalNameResolver(mlir::ModuleOp topLevel) {
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *topLevel.getBody()) {
    // Note that external modules *often* have name collisions, because they
    // correspond to the same verilog module with different parameters.
    if (isa<HWModuleExternOp>(op) || isa<HWModuleGeneratedOp>(op)) {
      auto name = getVerilogModuleNameAttr(&op).getValue();
      if (!sv::isNameValid(name))
        op.emitError("name \"")
            << name << "\" is not allowed in Verilog output";
      globalNameResolver.insertUsedName(name);
    }
  }

  // Legalize module and interface names.
  for (auto &op : *topLevel.getBody()) {
    // Legalize field names.
    fieldNameResolver.legalizeToplevelOperation(&op);
    if (auto module = dyn_cast<HWModuleOp>(op)) {
      legalizeModuleNames(module);
      continue;
    }

    // Legalize the name of the interface itself, as well as any signals and
    // modports within it.
    if (auto interface = dyn_cast<InterfaceOp>(op)) {
      legalizeInterfaceNames(interface);
      continue;
    }
  }
}

/// Check to see if the port names of the specified module conflict with
/// keywords or themselves.  If so, add the replacement names to
/// globalNameTable.
void GlobalNameResolver::legalizeModuleNames(HWModuleOp module) {
  MLIRContext *ctxt = module.getContext();
  // If the module's symbol itself conflicts, then set a "verilogName" attribute
  // on the module to reflect the name we need to use.
  StringRef oldName = module.getName();
  auto newName = globalNameResolver.getLegalName(oldName);
  if (newName != oldName)
    module->setAttr("verilogName", StringAttr::get(ctxt, newName));

  NameCollisionResolver nameResolver;
  auto verilogNameAttr = StringAttr::get(ctxt, "hw.verilogName");
  // Legalize the port names.
  size_t portIdx = 0;
  SmallVector<Attribute, 4> argNames, resultNames;
  for (const PortInfo &port : getAllModulePortInfos(module)) {
    auto newName = nameResolver.getLegalName(port.name);
    if (newName != port.name.getValue()) {
      globalNameTable.addRenamedPort(module, port, newName);
      if (port.isOutput())
        module.setResultAttr(port.argNum, verilogNameAttr,
                             StringAttr::get(ctxt, newName));
      else
        module.setArgAttr(port.argNum, verilogNameAttr,
                          StringAttr::get(ctxt, newName));
    }
    ++portIdx;
  }

  // Legalize the parameter names.
  for (auto param : module.parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto newName = nameResolver.getLegalName(paramAttr.getName());
    if (newName != paramAttr.getName().getValue())
      globalNameTable.addRenamedParam(module, paramAttr.getName(), newName);
  }

  // Legalize the value names.
  module.walk([&](Operation *op) {
    if (auto nameAttr = getDeclarationName(op)) {
      auto newName = nameResolver.getLegalName(nameAttr);
      if (newName != nameAttr.getValue()) {
        globalNameTable.addRenamedDeclaration(op, newName);
        op->setAttr(verilogNameAttr, StringAttr::get(ctxt, newName));
      }
    }
  });
}

void GlobalNameResolver::legalizeInterfaceNames(InterfaceOp interface) {
  auto newName = globalNameResolver.getLegalName(interface.getName());
  if (newName != interface.getName())
    globalNameTable.addRenamedInterfaceOp(interface, newName);

  NameCollisionResolver localNames;
  // Rename signals and modports.
  for (auto &op : *interface.getBodyBlock()) {
    if (isa<InterfaceSignalOp, InterfaceModportOp>(op)) {
      auto name = SymbolTable::getSymbolName(&op).getValue();
      auto newName = localNames.getLegalName(name);
      if (newName != name)
        globalNameTable.addRenamedInterfaceOp(&op, newName);
    }
  }
}

//===----------------------------------------------------------------------===//
// Public interface
//===----------------------------------------------------------------------===//

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable ExportVerilog::legalizeGlobalNames(ModuleOp topLevel) {
  GlobalNameResolver resolver(topLevel);
  return resolver.takeGlobalNameTable();
}
