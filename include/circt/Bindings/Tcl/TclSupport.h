//===- TclSupport.h - Reusable CIRCT Tcl support ----------------*- C++ -*-===//

#ifndef CIRCT_BINDINGS_TCL_TCLSUPPORT_H
#define CIRCT_BINDINGS_TCL_TCLSUPPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LogicalResult.h"

#include <tcl.h>

namespace circt::tcl {

class Environment;

mlir::LogicalResult initialize(Tcl_Interp *interp);
Environment *getEnvironment(Tcl_Interp *interp);
mlir::MLIRContext *getContext(Environment *environment);

Tcl_Obj *createOwnedModuleObject(Tcl_Interp *interp,
                                 mlir::OwningOpRef<mlir::ModuleOp> module);
mlir::FailureOr<mlir::ModuleOp> getModule(Tcl_Interp *interp,
                                          Tcl_Obj *object);
Tcl_Obj *createOperationHandle(Tcl_Interp *interp,
                               mlir::Operation *operation,
                               Tcl_Obj *owningModule);

int setError(Tcl_Interp *interp, const llvm::Twine &message);

} // namespace circt::tcl

#endif // CIRCT_BINDINGS_TCL_TCLSUPPORT_H
