//===- TclSupport.cpp - Reusable CIRCT Tcl support --------------*- C++ -*-===//

#include "circt/Bindings/Tcl/TclSupport.h"

#include "circt/InitAllDialects.h"
#include "mlir/IR/DialectRegistry.h"

#include <cstring>
#include <memory>
#include <string>

using namespace mlir;

namespace circt::tcl {

class Environment {
public:
  Environment() {
    DialectRegistry registry;
    circt::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
  }

  MLIRContext context;
};

namespace {

constexpr const char *environmentKey = "circt::tcl::Environment";

struct OwnedModuleRep {
  std::shared_ptr<OwningOpRef<ModuleOp>> module;
};

struct OperationHandleRep {
  Operation *operation = nullptr;
  Tcl_Obj *owningModule = nullptr;
};

int rejectConversion(Tcl_Interp *interp, Tcl_Obj *) {
  return setError(interp, "value is not a CIRCT operation handle");
}

void updateOwnedModuleString(Tcl_Obj *object) {
  auto *rep = static_cast<OwnedModuleRep *>(object->internalRep.otherValuePtr);
  std::string value;
  llvm::raw_string_ostream stream(value);
  (*rep->module)->print(stream);
  stream.flush();
  object->length = static_cast<int>(value.size());
  object->bytes = Tcl_Alloc(object->length + 1);
  memcpy(object->bytes, value.data(), value.size());
  object->bytes[object->length] = '\0';
}

void duplicateOwnedModule(Tcl_Obj *source, Tcl_Obj *duplicate) {
  auto *rep = static_cast<OwnedModuleRep *>(source->internalRep.otherValuePtr);
  duplicate->internalRep.otherValuePtr = new OwnedModuleRep{rep->module};
}

void freeOwnedModule(Tcl_Obj *object) {
  delete static_cast<OwnedModuleRep *>(object->internalRep.otherValuePtr);
}

void updateOperationHandleString(Tcl_Obj *object) {
  auto *rep =
      static_cast<OperationHandleRep *>(object->internalRep.otherValuePtr);
  std::string value;
  llvm::raw_string_ostream stream(value);
  rep->operation->print(stream);
  stream.flush();
  object->length = static_cast<int>(value.size());
  object->bytes = Tcl_Alloc(object->length + 1);
  memcpy(object->bytes, value.data(), value.size());
  object->bytes[object->length] = '\0';
}

void duplicateOperationHandle(Tcl_Obj *source, Tcl_Obj *duplicate) {
  auto *rep =
      static_cast<OperationHandleRep *>(source->internalRep.otherValuePtr);
  Tcl_IncrRefCount(rep->owningModule);
  duplicate->internalRep.otherValuePtr =
      new OperationHandleRep{rep->operation, rep->owningModule};
}

void freeOperationHandle(Tcl_Obj *object) {
  auto *rep =
      static_cast<OperationHandleRep *>(object->internalRep.otherValuePtr);
  Tcl_DecrRefCount(rep->owningModule);
  delete rep;
}

Tcl_ObjType ownedModuleType = {
    "CirctOwnedModule", freeOwnedModule, duplicateOwnedModule,
    updateOwnedModuleString, rejectConversion};
Tcl_ObjType operationHandleType = {
    "CirctOperationHandle", freeOperationHandle, duplicateOperationHandle,
    updateOperationHandleString, rejectConversion};

void deleteEnvironment(ClientData data, Tcl_Interp *) {
  delete static_cast<Environment *>(data);
}

} // namespace

LogicalResult initialize(Tcl_Interp *interp) {
  if (getEnvironment(interp))
    return success();
  Tcl_RegisterObjType(&ownedModuleType);
  Tcl_RegisterObjType(&operationHandleType);
  Tcl_SetAssocData(interp, environmentKey, deleteEnvironment,
                   new Environment());
  return success();
}

Environment *getEnvironment(Tcl_Interp *interp) {
  return static_cast<Environment *>(
      Tcl_GetAssocData(interp, environmentKey, nullptr));
}

MLIRContext *getContext(Environment *environment) {
  return environment ? &environment->context : nullptr;
}

Tcl_Obj *createOwnedModuleObject(Tcl_Interp *, OwningOpRef<ModuleOp> module) {
  auto *object = Tcl_NewObj();
  object->typePtr = &ownedModuleType;
  object->internalRep.otherValuePtr = new OwnedModuleRep{
      std::make_shared<OwningOpRef<ModuleOp>>(std::move(module))};
  Tcl_InvalidateStringRep(object);
  return object;
}

FailureOr<ModuleOp> getModule(Tcl_Interp *interp, Tcl_Obj *object) {
  if (object->typePtr != &ownedModuleType) {
    setError(interp, "expected a CIRCT owned module value");
    return failure();
  }
  auto *rep = static_cast<OwnedModuleRep *>(object->internalRep.otherValuePtr);
  if (!rep || !rep->module || !*rep->module) {
    setError(interp, "CIRCT module value is no longer valid");
    return failure();
  }
  return **rep->module;
}

Tcl_Obj *createOperationHandle(Tcl_Interp *, Operation *operation,
                               Tcl_Obj *owningModule) {
  auto *object = Tcl_NewObj();
  Tcl_IncrRefCount(owningModule);
  object->typePtr = &operationHandleType;
  object->internalRep.otherValuePtr =
      new OperationHandleRep{operation, owningModule};
  Tcl_InvalidateStringRep(object);
  return object;
}

int setError(Tcl_Interp *interp, const llvm::Twine &message) {
  std::string text = message.str();
  Tcl_SetObjResult(interp,
                   Tcl_NewStringObj(text.data(), static_cast<int>(text.size())));
  return TCL_ERROR;
}

} // namespace circt::tcl
