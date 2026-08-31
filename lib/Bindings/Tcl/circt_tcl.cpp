#include "circt/Bindings/Tcl/TclSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <system_error>

using namespace mlir;

namespace {

int loadMlir(ClientData, Tcl_Interp *interp, int objc,
             Tcl_Obj *const objv[]) {
  if (objc != 2) {
    Tcl_WrongNumArgs(interp, 1, objv, "path");
    return TCL_ERROR;
  }

  auto *environment = circt::tcl::getEnvironment(interp);
  std::string errorMessage;
  auto input = mlir::openInputFile(Tcl_GetString(objv[1]), &errorMessage);
  if (!input)
    return circt::tcl::setError(interp, errorMessage);

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagnosticHandler(
      sourceMgr, circt::tcl::getContext(environment));
  auto module = parseSourceFile<ModuleOp>(
      sourceMgr, circt::tcl::getContext(environment));
  if (!module)
    return circt::tcl::setError(interp, "error loading module");
  Tcl_SetObjResult(interp,
                   circt::tcl::createOwnedModuleObject(interp,
                                                       std::move(module)));
  return TCL_OK;
}

int saveMlir(ClientData, Tcl_Interp *interp, int objc,
             Tcl_Obj *const objv[]) {
  if (objc != 3) {
    Tcl_WrongNumArgs(interp, 1, objv, "module path");
    return TCL_ERROR;
  }
  auto module = circt::tcl::getModule(interp, objv[1]);
  if (failed(module))
    return TCL_ERROR;
  std::error_code error;
  llvm::raw_fd_ostream output(Tcl_GetString(objv[2]), error);
  if (error)
    return circt::tcl::setError(interp, error.message());
  (*module).print(output);
  output << '\n';
  return TCL_OK;
}

int compatibilityCommand(ClientData data, Tcl_Interp *interp, int objc,
                         Tcl_Obj *const objv[]) {
  if (objc == 4 && !strcmp(Tcl_GetString(objv[1]), "load") &&
      !strcmp(Tcl_GetString(objv[2]), "MLIR")) {
    Tcl_Obj *arguments[] = {objv[0], objv[3]};
    return loadMlir(data, interp, 2, arguments);
  }
  return circt::tcl::setError(interp, "usage: circt load MLIR path");
}

} // namespace

extern "C" {

int DLLEXPORT Circt_Init(Tcl_Interp *interp) {
  if (!Tcl_InitStubs(interp, TCL_VERSION, 0))
    return TCL_ERROR;
  if (failed(circt::tcl::initialize(interp)))
    return TCL_ERROR;
  if (Tcl_PkgProvide(interp, "Circt", "1.0") == TCL_ERROR)
    return TCL_ERROR;

  Tcl_CreateNamespace(interp, "circt", nullptr, nullptr);
  Tcl_CreateObjCommand(interp, "circt::load_mlir", loadMlir, nullptr, nullptr);
  Tcl_CreateObjCommand(interp, "circt::save_mlir", saveMlir, nullptr, nullptr);
  Tcl_CreateObjCommand(interp, "circt", compatibilityCommand, nullptr,
                       nullptr);
  return TCL_OK;
}
}
