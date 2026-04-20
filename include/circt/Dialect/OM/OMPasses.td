//===- OMPasses.td - OM dialect passes ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def FreezePaths: Pass<"om-freeze-paths", "mlir::ModuleOp"> {
  let summary = "Hard code all path information";
  let description = [{
    Replaces paths to hardware with hard-coded string paths.  This pass should
    only run once the hierarchy will no longer change, and the final names for
    objects have been decided.
  }];
  let constructor = "circt::om::createFreezePathsPass()";
}

def LinkModules: Pass<"om-link-modules", "mlir::ModuleOp"> {
  let summary = "Link separated OM modules into a single module";
  let description = [{
    Flatten nested modules and resolve external classes.
  }];
  let constructor = "circt::om::createOMLinkModulesPass()";
}

def VerifyObjectFields: Pass<"om-verify-object-fields"> {
  let summary = "Verify fields of ObjectOp are valid";
  let description = [{
    Verify object fields are valid.
  }];
  let constructor = "circt::om::createVerifyObjectFieldsPass()";
}

def ElaborateObject: Pass<"om-elaborate-object", "mlir::ModuleOp"> {
  let summary = "Elaborate zero-input OM classes into om.elaborated_object form";
  let description = [{
    Creates synthetic wrapper classes for zero-input OM classes and runs the
    shared OM elaboration transform on those wrappers.
  }];
  let options = [
    Option<"targetClass", "target-class", "std::string", /*default=*/"",
           "Only create and elaborate a wrapper for the named zero-input class">
  ];
  let constructor = "circt::om::createElaborateObjectPass()";
}

def StripOMPass : Pass<"strip-om", "mlir::ModuleOp"> {
  let summary = "Remove OM information";
  let description = [{
    Removes all OM operations from the IR. Useful as a prepass in pipelines that
    have to discard OM.
  }];
}
