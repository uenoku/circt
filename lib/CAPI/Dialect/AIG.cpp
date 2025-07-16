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
#include "circt/Support/InstanceGraph.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <tuple>

using namespace circt;
using namespace circt::aig;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIG, aig, circt::aig::AIGDialect)

void registerAIGPasses() { circt::aig::registerPasses(); }

// Wrapper struct to hold both the analysis and the analysis manager
struct LongestPathAnalysisWrapper {
  std::unique_ptr<mlir::ModuleAnalysisManager> analysisManager;
  std::unique_ptr<LongestPathAnalysis> analysis;
};

struct LongestPathObjectWrapper {
  llvm::PointerUnion<Object *, DataflowPath::OutputPort *> object;
};

DEFINE_C_API_PTR_METHODS(AIGLongestPathAnalysis, LongestPathAnalysisWrapper)
DEFINE_C_API_PTR_METHODS(AIGLongestPathCollection, LongestPathCollection)
DEFINE_C_API_PTR_METHODS(AIGLongestPathDataflowPath, DataflowPath)
DEFINE_C_API_PTR_METHODS(AIGLongestPathHistory,
                         llvm::ImmutableListImpl<DebugPoint>)
DEFINE_C_API_PTR_METHODS(HWInstancePath, circt::igraph::InstancePath)

LongestPathObjectWrapper unwrap(AIGLongestPathObject object) {
  LongestPathObjectWrapper wrapper;
  wrapper.object = llvm::PointerUnion<
      Object *, DataflowPath::OutputPort *>::getFromOpaqueValue(object.ptr);
  return wrapper;
}

AIGLongestPathObject wrap(LongestPathObjectWrapper object) {
  return AIGLongestPathObject{object.object.getOpaqueValue()};
}

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

// ===----------------------------------------------------------------------===//
// DataflowPath
// ===----------------------------------------------------------------------===//

AIGLongestPathDataflowPath
aigLongestPathCollectionGetPath(AIGLongestPathCollection collection,
                                size_t index) {
  auto *wrapper = unwrap(collection);
  auto &path = wrapper->paths[index];
  return wrap(&path);
}

// ===----------------------------------------------------------------------===//
// DataflowPath
// ===----------------------------------------------------------------------===//

int64_t aigLongestPathDataflowPathGetDelay(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrapper->getDelay();
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanIn(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  auto &fanIn = wrapper->getFanIn();
  AIGLongestPathObject object;
  object.ptr = const_cast<Object *>(&fanIn);
  return object;
}

AIGLongestPathObject
aigLongestPathDataflowPathGetFanOut(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  AIGLongestPathObject result;
  if (auto *object = std::get_if<Object>(&wrapper->getFanOut())) {
    result.ptr = const_cast<Object *>(object);
  } else {
    auto *ptr = std::get_if<DataflowPath::OutputPort>(&wrapper->getFanOut());
    result.ptr = const_cast<DataflowPath::OutputPort *>(ptr);
  }
  return result;
}

AIGLongestPathHistory
aigLongestPathDataflowPathGetHistory(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrap(const_cast<llvm::ImmutableListImpl<DebugPoint> *>(
      wrapper->getHistory().getInternalPointer()));
}

// ===----------------------------------------------------------------------===//
// History
// ===----------------------------------------------------------------------===//

bool aigLongestPathHistoryIsEmpty(AIGLongestPathHistory history) {
  auto *wrapper = unwrap(history);
  return llvm::ImmutableList<DebugPoint>(wrapper).isEmpty();
}

void aigLongestPathHistoryGetHead(AIGLongestPathHistory history,
                                  AIGLongestPathObject *object, int64_t *delay,
                                  MlirStringRef *comment) {
  auto *wrapper = unwrap(history);
  auto list = llvm::ImmutableList<DebugPoint>(wrapper);

  auto &head = list.getHead();
  object->ptr = const_cast<Object *>(&head.object);
  *delay = head.delay;
  *comment = mlirStringRefCreate(head.comment.data(), head.comment.size());
}

AIGLongestPathHistory
aigLongestPathHistoryGetTail(AIGLongestPathHistory history) {
  auto *wrapper = unwrap(history);
  auto list = llvm::ImmutableList<DebugPoint>(wrapper);
  auto tail = list.getTail().getInternalPointer();
  return wrap(const_cast<llvm::ImmutableListImpl<DebugPoint> *>(tail));
}

// ===----------------------------------------------------------------------===//
// InstancePath
// ===----------------------------------------------------------------------===//

size_t hwInstancePathSize(HWInstancePath instancePath) {
  auto *wrapper = unwrap(instancePath);
  return wrapper->size();
}

void hwInstancePathGet(HWInstancePath instancePath, size_t index,
                       MlirOperation *instance) {
  auto *wrapper = unwrap(instancePath);
  auto path = wrapper->getPath();
  assert(wrapper->size() == index);
  llvm::errs() << "Instance " << index << "\n";

  for (size_t i = 0; i < index; i++) {
    llvm::errs() << "Instance " << i << ": " << path[i] << "\n";
    auto inst = path[i];
    instance[i] = wrap(inst);
  }
}

MlirOperation
aigLongestPathDataflowPathGetRoot(AIGLongestPathDataflowPath path) {
  auto *wrapper = unwrap(path);
  return wrap(wrapper->getRoot());
}

HWInstancePath
aigLongestPathObjectGetInstancePath(AIGLongestPathObject object) {
  auto wrapper = unwrap(object);
  auto *ptr = wrapper.object.dyn_cast<Object *>();
  if (ptr)
    return wrap(&ptr->instancePath);

  return {nullptr};
}

MlirStringRef aigLongestPathObjectName(AIGLongestPathObject object) {
  auto wrapper = unwrap(object);
  auto *ptr = wrapper.object.dyn_cast<Object *>();

  if (ptr) {
    auto name = getNameForValue(ptr->value);
    return mlirStringRefCreate(name.data(), name.size());
  }
  assert(wrapper.object.is<DataflowPath::OutputPort *>());

  auto [module, resultNumber, bitPos] =
      *wrapper.object.dyn_cast<DataflowPath::OutputPort *>();
  auto name = module.getOutputName(resultNumber);
  return mlirStringRefCreate(name.data(), name.size());
}

size_t aigLongestPathObjectBitPos(AIGLongestPathObject object) {
  auto wrapper = unwrap(object);
  auto *ptr = wrapper.object.dyn_cast<Object *>();
  if (ptr)
    return ptr->bitPos;

  auto [module, resultNumber, bitPos] =
      *wrapper.object.dyn_cast<DataflowPath::OutputPort *>();
  return bitPos;
}
