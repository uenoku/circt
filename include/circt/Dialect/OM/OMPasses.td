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
}

def ElaborateObject: Pass<"om-elaborate-object", "mlir::ModuleOp"> {
  let summary = "Elaborate a OM object into om.elaborated_object form";
  let description = [{
    Elaborates a specified zero-input OM class by inlining all object
    instantiations and converting them to om.elaborated_object form.
    The transformation happens in-place within the specified class.

    If test is true, elaborates all nullary classes in the module
    for testing purposes.
  }];
  let options = [
    Option<"targetClass", "target-class", "std::string", /*default=*/"",
           "The class to elaborate">,
    Option<"test", "test", "bool", /*default=*/"false",
           "Test mode: elaborate all nullary classes in the module">
  ];
}

def StripOMPass : Pass<"strip-om", "mlir::ModuleOp"> {
  let summary = "Remove OM information";
  let description = [{
    Removes all OM operations from the IR. Useful as a prepass in pipelines that
    have to discard OM.
  }];
}
