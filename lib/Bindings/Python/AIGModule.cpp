//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Dialect/AIG.h"

#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/Wrap.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
#include <string_view>

namespace nb = nanobind;

using namespace circt;
using namespace mlir::python::nanobind_adaptors;

/// Populate the aig python module.
void circt::python::populateDialectAIGSubmodule(nb::module_ &m) {
  m.doc() = "AIG dialect Python native extension";

  // LongestPathAnalysis class
  nb::class_<AIGLongestPathAnalysis>(m, "_LongestPathAnalysis")
      .def(
          "__init__",
          [](AIGLongestPathAnalysis *self, MlirOperation module,
             bool traceDebugPoints) {
            new (self) AIGLongestPathAnalysis(
                aigLongestPathAnalysisCreate(module, traceDebugPoints));
          },
          nb::arg("module"), nb::arg("trace_debug_points") = true)
      .def("__del__",
           [](AIGLongestPathAnalysis &self) {
             aigLongestPathAnalysisDestroy(self);
           })
      .def(
          "get_all_paths",
          [](AIGLongestPathAnalysis *self, const std::string &moduleName,
             const std::string &fanoutFilter, const std::string &faninFilter,
             bool elaboratePaths) -> AIGLongestPathCollection {
            MlirStringRef moduleNameRef =
                mlirStringRefCreateFromCString(moduleName.c_str());
            MlirStringRef fanoutFilterRef =
                mlirStringRefCreateFromCString(fanoutFilter.c_str());
            MlirStringRef faninFilterRef =
                mlirStringRefCreateFromCString(faninFilter.c_str());

            if (aigLongestPathCollectionIsNull(
                    aigLongestPathAnalysisGetAllPaths(
                        *self, moduleNameRef, fanoutFilterRef, faninFilterRef,
                        elaboratePaths)))
              throw nb::value_error(
                  "Failed to get all paths, see previous error(s).");

            auto collection =
                AIGLongestPathCollection(aigLongestPathAnalysisGetAllPaths(
                    *self, moduleNameRef, fanoutFilterRef, faninFilterRef,
                    elaboratePaths));
            return collection;
          },
          nb::arg("module_name"), nb::arg("fanout_filter") = "",
          nb::arg("fanin_filter") = "", nb::arg("elaborate_paths") = true);

  nb::class_<AIGLongestPathCollection>(m, "_LongestPathCollection")
      .def("__del__",
           [](AIGLongestPathCollection &self) {
             aigLongestPathCollectionDestroy(self);
           })
      .def("get_size",
           [](AIGLongestPathCollection &self) {
             return aigLongestPathCollectionGetSize(self);
           })
      .def("get_path",
           [](AIGLongestPathCollection &self,
              int pathIndex) -> AIGLongestPathDataflowPath {
             return aigLongestPathCollectionGetPath(self, pathIndex);
           })
      .def(
          "_diff",
          [](AIGLongestPathCollection &self, AIGLongestPathCollection &other)
              -> std::tuple<AIGLongestPathCollection, AIGLongestPathCollection,
                            AIGLongestPathCollection,
                            AIGLongestPathCollection> {
            AIGLongestPathCollection uniqueLhs, uniqueRhs, differentLhs,
                differentRhs;
            if (!aigLongestPathCollectionDiff(self, other, &uniqueLhs,
                                              &uniqueRhs, &differentLhs,
                                              &differentRhs))
              throw nb::value_error(
                  "Failed to diff collections, see previous error(s).");
            return std::make_tuple(uniqueLhs, uniqueRhs, differentLhs,
                                   differentRhs);
          },
          nb::arg("other"));
  nb::class_<AIGLongestPathDataflowPath>(m, "_LongestPathDataflowPath")
      .def("get_delay",
           [](AIGLongestPathDataflowPath &self) {
             return aigLongestPathDataflowPathGetDelay(self);
           })
      .def("get_fan_in",
           [](AIGLongestPathDataflowPath &self) {
             return aigLongestPathDataflowPathGetFanIn(self);
           })
      .def("get_fan_out",
           [](AIGLongestPathDataflowPath &self) {
             return aigLongestPathDataflowPathGetFanOut(self);
           })
      .def("get_history", [](AIGLongestPathDataflowPath &self) {
        return aigLongestPathDataflowPathGetHistory(self);
      });
}
