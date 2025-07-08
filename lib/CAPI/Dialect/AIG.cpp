//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/AIG.h"
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/AIG/Analysis/LongestPathAnalysis.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace circt::aig;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIG, aig, circt::aig::AIGDialect)

void registerAIGPasses() { circt::aig::registerPasses(); }

// Wrapper struct to hold both the analysis and the analysis manager
struct LongestPathAnalysisWrapper {
  std::unique_ptr<mlir::ModuleAnalysisManager> analysisManager;
  std::unique_ptr<LongestPathAnalysis> analysis;
};

DEFINE_C_API_PTR_METHODS(AIGLongestPathAnalysis, LongestPathAnalysisWrapper)
DEFINE_C_API_PTR_METHODS(AIGLongestPathCollection, LongestPathCollection)
DEFINE_C_API_PTR_METHODS(AIGLongestPathDataflowPath, DataflowPath)
DEFINE_C_API_PTR_METHODS(AIGLongestPathObject, DataflowPath::FanOutType)
DEFINE_C_API_PTR_METHODS(AIGLongestPathHistory, llvm::ImmutableList<DebugPoint>)

//===----------------------------------------------------------------------===//
// LongestPathAnalysis C API
//===----------------------------------------------------------------------===//

AIGLongestPathAnalysis aigLongestPathAnalysisCreate(MlirOperation module,
                                                    bool traceDebugPoints) {
  auto *op = unwrap(module);
  auto *wrapper = new LongestPathAnalysisWrapper();
  wrapper->analysisManager =
      std::make_unique<mlir::ModuleAnalysisManager>(op, nullptr);
  mlir::AnalysisManager am = *wrapper->analysisManager;
  if (traceDebugPoints)
    wrapper->analysis = std::make_unique<LongestPathAnalysisWithTrace>(op, am);
  else
    wrapper->analysis = std::make_unique<LongestPathAnalysis>(op, am);
  return wrap(wrapper);
}

void aigLongestPathAnalysisDestroy(AIGLongestPathAnalysis analysis) {
  delete unwrap(analysis);
}

AIGLongestPathCollection aigLongestPathAnalysisGetAllPaths(
    AIGLongestPathAnalysis analysis, MlirStringRef moduleName,
    MlirStringRef fanoutFilter, MlirStringRef faninFilter,
    bool elaboratePaths) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto moduleNameAttr = StringAttr::get(lpa->getContext(), unwrap(moduleName));

  auto *collection = new LongestPathCollection(lpa->getContext());
  if (!lpa->isAnalysisAvailable(moduleNameAttr) ||
      failed(
          lpa->getAllPaths(moduleNameAttr, collection->paths, elaboratePaths)))
    return {nullptr};

  collection->sortInDescendingOrder();
  collection->filterByFanOut(unwrap(fanoutFilter));
  collection->filterByFanIn(unwrap(faninFilter));
  return wrap(collection);
}

// ===----------------------------------------------------------------------===//
// LongestPathCollection
// ===----------------------------------------------------------------------===//

bool aigLongestPathCollectionIsNull(AIGLongestPathCollection collection) {
  return !collection.ptr;
}

void aigLongestPathCollectionDestroy(AIGLongestPathCollection collection) {
  delete unwrap(collection);
}

size_t aigLongestPathCollectionGetSize(AIGLongestPathCollection collection) {
  auto *wrapper = unwrap(collection);
  return wrapper->paths.size();
}

AIGLongestPathDataflowPath
aigLongestPathCollectionGetPath(AIGLongestPathCollection collection,
                                int pathIndex) {
  auto *wrapper = unwrap(collection);

  // Check if pathIndex is valid
  if (pathIndex < 0 || pathIndex >= static_cast<int>(wrapper->paths.size()))
    return {nullptr};

  AIGLongestPathDataflowPath path;
  // It's safe to const_cast here because the C API does not allow
  // modification of the path.
  path.ptr = const_cast<DataflowPath *>(&wrapper->getPath(pathIndex));
  return path;
}

MlirStringRef
aigLongestPathCollectionGetPathAsJson(AIGLongestPathCollection collection,
                                      int pathIndex) {
  auto *wrapper = unwrap(collection);

  // Check if pathIndex is valid
  if (pathIndex < 0 || pathIndex >= static_cast<int>(wrapper->paths.size()))
    return wrap(llvm::StringRef(""));

  // Convert the specific path to JSON
  // FIXME: Avoid converting to JSON and then back to string. Use native
  // CAPI instead once data structure is stabilized.
  llvm::json::Value pathJson = toJSON(wrapper->paths[pathIndex]);

  std::string jsonStr;
  llvm::raw_string_ostream os(jsonStr);
  os << pathJson;

  auto ctx = wrap(wrapper->getContext());

  // Use MLIR StringAttr to manage the string lifetime.
  // FIXME: This is safe but expensive. Consider manually managing the string
  // lifetime.
  MlirAttribute strAttr =
      mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(os.str().c_str()));
  return mlirStringAttrGetValue(strAttr);
}

bool aigLongestPathCollectionDiff(AIGLongestPathCollection lhs,
                                  AIGLongestPathCollection rhs,
                                  AIGLongestPathCollection *uniqueLhs,
                                  AIGLongestPathCollection *uniqueRhs,
                                  AIGLongestPathCollection *differentLhs,
                                  AIGLongestPathCollection *differentRhs) {
  auto *lhsCollection = unwrap(lhs);
  auto *rhsCollection = unwrap(rhs);
  Difference diff(*lhsCollection, *rhsCollection);

  if (uniqueLhs)
    *uniqueLhs = wrap(diff.lhsUniquePaths.release());
  if (uniqueRhs)
    *uniqueRhs = wrap(diff.rhsUniquePaths.release());
  if (differentLhs)
    *differentLhs = wrap(diff.lhsDifferentDelay.release());
  if (differentRhs)
    *differentRhs = wrap(diff.rhsDifferentDelay.release());

  return true;
}

//===----------------------------------------------------------------------===//
// DataflowPath C API
//===----------------------------------------------------------------------===//

int64_t
aigLongestPathDataflowPathGetDelay(AIGLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  if (!path)
    return -1;
  return path->getDelay();
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanIn(AIGLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  if (!path)
    return {nullptr};

  // Return a pointer to the fanIn object
  return wrap(const_cast<Object *>(&path->getFanIn()));
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanOut(AIGLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  if (!path)
    return {nullptr};

  // Check if fanOut is an Object (not an output port)
  const auto &fanOut = path->getFanOut();
  if (auto *object = std::get_if<Object>(&fanOut)) {
    return wrap(const_cast<Object *>(object));
  }

  // If it's an output port, we can't return it as an Object
  // Return null to indicate this is not an Object
  return {nullptr};
}

AIGLongestPathHistory
aigLongestPathDataflowPathGetHistory(AIGLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  if (!path)
    return {nullptr};

  // Return a pointer to the history list
  return wrap(
      const_cast<llvm::ImmutableList<DebugPoint> *>(&path->getHistory()));
}

//===----------------------------------------------------------------------===//
// LongestPathHistory C API
//===----------------------------------------------------------------------===//

size_t aigLongestPathHistoryGetSize(AIGLongestPathHistory history) {
  auto *historyList = unwrap(history);
  if (!historyList)
    return 0;

  // Count the elements in the immutable list
  size_t count = 0;
  for (auto it = historyList->begin(); it != historyList->end(); ++it) {
    count++;
  }
  return count;
}

void aigLongestPathHistoryGet(AIGLongestPathHistory history, size_t index,
                              AIGLongestPathObject *object, int64_t *delay,
                              MlirAttribute *comment) {
  auto *historyList = unwrap(history);
  if (!historyList) {
    if (object)
      *object = {nullptr};
    if (delay)
      *delay = -1;
    if (comment)
      *comment = {nullptr};
    return;
  }

  // Find the element at the given index
  size_t currentIndex = 0;
  for (auto it = historyList->begin(); it != historyList->end();
       ++it, ++currentIndex) {
    if (currentIndex == index) {
      const DebugPoint &debugPoint = *it;

      if (object) {
        *object = wrap(const_cast<Object *>(&debugPoint.object));
      }

      if (delay) {
        *delay = debugPoint.delay;
      }

      if (comment) {
        // Convert StringRef to MlirAttribute (StringAttr)
        auto *ctx = debugPoint.object.value.getContext();
        auto strAttr = StringAttr::get(ctx, debugPoint.comment);
        *comment = wrap(static_cast<Attribute>(strAttr));
      }

      return;
    }
  }

  // Index out of bounds
  if (object)
    *object = {nullptr};
  if (delay)
    *delay = -1;
  if (comment)
    *comment = {nullptr};
}