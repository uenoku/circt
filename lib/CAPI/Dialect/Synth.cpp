//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Synth.h"

#include "circt-c/Support/InstanceGraph.h"
#include "circt/Dialect/Synth/Analysis/LongestPathAnalysis.h"
#include "circt/Dialect/Synth/SynthDialect.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Dialect/Synth/Transforms/SynthesisPipeline.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <tuple>

using namespace circt;
using namespace circt::synth;

void registerSynthesisPipeline() { circt::synth::registerSynthesisPipeline(); }

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Synth, synth, circt::synth::SynthDialect)

void registerSynthPasses() { circt::synth::registerPasses(); }

// Wrapper struct to hold both the analysis and the analysis manager
struct LongestPathAnalysisWrapper {
  std::unique_ptr<mlir::ModuleAnalysisManager> analysisManager;
  std::unique_ptr<LongestPathAnalysis> analysis;
};

// Wrapper struct to hold both the collection and the analysis for CAPI
struct LongestPathCollectionWrapper {
  std::unique_ptr<LongestPathCollection> collection;
  const LongestPathAnalysis *analysis;
};

DEFINE_C_API_PTR_METHODS(SynthLongestPathAnalysis, LongestPathAnalysisWrapper)
DEFINE_C_API_PTR_METHODS(SynthLongestPathCollection, LongestPathCollectionWrapper)
DEFINE_C_API_PTR_METHODS(SynthLongestPathDataflowPath, DataflowPath)
DEFINE_C_API_PTR_METHODS(SynthLongestPathHistory, SmallVector<DebugPoint>)

// SynthLongestPathObject is a pointer to either an Object or an OutputPort so
// we can not use DEFINE_C_API_PTR_METHODS.
llvm::PointerUnion<Object *, DataflowPath::OutputPort *>
unwrap(SynthLongestPathObject object) {
  return llvm::PointerUnion<
      Object *, DataflowPath::OutputPort *>::getFromOpaqueValue(object.ptr);
}

SynthLongestPathObject
wrap(llvm::PointerUnion<Object *, DataflowPath::OutputPort *> object) {
  return SynthLongestPathObject{object.getOpaqueValue()};
}

SynthLongestPathObject wrap(const Object *object) {
  auto ptr = llvm::PointerUnion<Object *, DataflowPath::OutputPort *>(
      const_cast<Object *>(object));
  return wrap(ptr);
}

SynthLongestPathObject wrap(const DataflowPath::OutputPort *object) {
  auto ptr = llvm::PointerUnion<Object *, DataflowPath::OutputPort *>(
      const_cast<DataflowPath::OutputPort *>(object));
  return wrap(ptr);
}

//===----------------------------------------------------------------------===//
// LongestPathAnalysis C API
//===----------------------------------------------------------------------===//

SynthLongestPathAnalysis
synthLongestPathAnalysisCreate(MlirOperation module, bool collectDebugInfo,
                               bool keepOnlyMaxDelayPaths, bool lazyComputation,
                               MlirStringRef topModuleName) {
  auto *op = unwrap(module);
  auto *wrapper = new LongestPathAnalysisWrapper();
  wrapper->analysisManager =
      std::make_unique<mlir::ModuleAnalysisManager>(op, nullptr);
  mlir::AnalysisManager am = *wrapper->analysisManager;
  auto topModuleNameAttr =
      StringAttr::get(op->getContext(), unwrap(topModuleName));
  wrapper->analysis = std::make_unique<LongestPathAnalysis>(
      op, am,
      LongestPathAnalysisOptions(lazyComputation,
                                 keepOnlyMaxDelayPaths, topModuleNameAttr));
  return wrap(wrapper);
}

void synthLongestPathAnalysisDestroy(SynthLongestPathAnalysis analysis) {
  delete unwrap(analysis);
}

SynthLongestPathCollection
synthLongestPathAnalysisGetPaths(SynthLongestPathAnalysis analysis,
                                 MlirValue value, int64_t bitPos,
                                 bool elaboratePaths) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto *collWrapper = new LongestPathCollectionWrapper();
  collWrapper->collection = std::make_unique<LongestPathCollection>(lpa->getContext());
  collWrapper->analysis = lpa;
  auto result =
      lpa->computeGlobalPaths(unwrap(value), bitPos, collWrapper->collection->paths);
  if (failed(result))
    return {nullptr};
  collWrapper->collection->sortInDescendingOrder();
  return wrap(collWrapper);
}

SynthLongestPathCollection
synthLongestPathAnalysisGetInternalPaths(SynthLongestPathAnalysis analysis,
                                         MlirStringRef moduleName,
                                         bool elaboratePaths) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto moduleNameAttr = StringAttr::get(lpa->getContext(), unwrap(moduleName));

  auto *collWrapper = new LongestPathCollectionWrapper();
  collWrapper->collection = std::make_unique<LongestPathCollection>(lpa->getContext());
  collWrapper->analysis = lpa;
  if (!lpa->isAnalysisAvailable(moduleNameAttr) ||
      failed(lpa->getInternalPaths(moduleNameAttr, collWrapper->collection->paths,
                                   elaboratePaths)))
    return {nullptr};

  collWrapper->collection->sortInDescendingOrder();
  return wrap(collWrapper);
}

// Get external paths from module input ports to internal sequential elements.
SynthLongestPathCollection
synthLongestPathAnalysisGetPathsFromInputPortsToInternal(
    SynthLongestPathAnalysis analysis, MlirStringRef moduleName) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto moduleNameAttr = StringAttr::get(lpa->getContext(), unwrap(moduleName));

  auto *collWrapper = new LongestPathCollectionWrapper();
  collWrapper->collection = std::make_unique<LongestPathCollection>(lpa->getContext());
  collWrapper->analysis = lpa;
  if (!lpa->isAnalysisAvailable(moduleNameAttr) ||
      failed(lpa->getOpenPathsFromInputPortsToInternal(moduleNameAttr,
                                                       collWrapper->collection->paths)))
    return {nullptr};

  collWrapper->collection->sortInDescendingOrder();
  return wrap(collWrapper);
}

// Get external paths from internal sequential elements to module output ports.
SynthLongestPathCollection
synthLongestPathAnalysisGetPathsFromInternalToOutputPorts(
    SynthLongestPathAnalysis analysis, MlirStringRef moduleName) {
  auto *wrapper = unwrap(analysis);
  auto *lpa = wrapper->analysis.get();
  auto moduleNameAttr = StringAttr::get(lpa->getContext(), unwrap(moduleName));

  auto *collWrapper = new LongestPathCollectionWrapper();
  collWrapper->collection = std::make_unique<LongestPathCollection>(lpa->getContext());
  collWrapper->analysis = lpa;
  if (!lpa->isAnalysisAvailable(moduleNameAttr) ||
      failed(lpa->getOpenPathsFromInternalToOutputPorts(moduleNameAttr,
                                                        collWrapper->collection->paths)))
    return {nullptr};

  collWrapper->collection->sortInDescendingOrder();
  return wrap(collWrapper);
}

// ===----------------------------------------------------------------------===//
// LongestPathCollection
// ===----------------------------------------------------------------------===//

bool synthLongestPathCollectionIsNull(SynthLongestPathCollection collection) {
  return !collection.ptr;
}

void synthLongestPathCollectionDestroy(SynthLongestPathCollection collection) {
  delete unwrap(collection);
}

size_t
synthLongestPathCollectionGetSize(SynthLongestPathCollection collection) {
  auto *wrapper = unwrap(collection);
  return wrapper->collection->paths.size();
}

// Get a specific path from the collection as DataflowPath object
SynthLongestPathDataflowPath
synthLongestPathCollectionGetDataflowPath(SynthLongestPathCollection collection,
                                          size_t index) {
  auto *collWrapper = unwrap(collection);
  return wrap(&collWrapper->collection->paths[index]);
}

void synthLongestPathCollectionMerge(SynthLongestPathCollection dest,
                                     SynthLongestPathCollection src) {
  auto *destWrapper = unwrap(dest);
  auto *srcWrapper = unwrap(src);
  destWrapper->collection->merge(*srcWrapper->collection);
}

void synthLongestPathCollectionDropNonCriticalPaths(
    SynthLongestPathCollection collection, bool perEndPoint) {
  auto *wrapper = unwrap(collection);
  wrapper->collection->dropNonCriticalPaths(perEndPoint);
}

SynthLongestPathHistory
synthLongestPathCollectionGetHistory(SynthLongestPathCollection collection,
                                     SynthLongestPathDataflowPath dataflowPath) {
  auto *collWrapper = unwrap(collection);
  auto *path = unwrap(dataflowPath);
  auto *history = new SmallVector<DebugPoint>();

  if (collWrapper->analysis) {
    path->getHistory(*collWrapper->analysis, *history);
  }

  return wrap(history);
}

//===----------------------------------------------------------------------===//
// DataflowPath
//===----------------------------------------------------------------------===//

int64_t
synthLongestPathDataflowPathGetDelay(SynthLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  return path->getDelay();
}

SynthLongestPathObject synthLongestPathDataflowPathGetStartPoint(
    SynthLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  auto &startPoint = path->getStartPoint();
  return wrap(const_cast<Object *>(&startPoint));
}

SynthLongestPathObject
synthLongestPathDataflowPathGetEndPoint(SynthLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  if (auto *object = std::get_if<Object>(&path->getEndPoint())) {
    return wrap(object);
  }
  auto *ptr = std::get_if<DataflowPath::OutputPort>(&path->getEndPoint());
  return wrap(ptr);
}

SynthLongestPathHistory
synthLongestPathDataflowPathGetHistory(SynthLongestPathAnalysis analysis,
                                       SynthLongestPathDataflowPath dataflowPath) {
  auto *analysisWrapper = unwrap(analysis);
  auto *path = unwrap(dataflowPath);
  auto *history = new SmallVector<DebugPoint>();

  if (analysisWrapper && analysisWrapper->analysis) {
    path->getHistory(*analysisWrapper->analysis, *history);
  }

  return wrap(history);
}

MlirOperation
synthLongestPathDataflowPathGetRoot(SynthLongestPathDataflowPath dataflowPath) {
  auto *path = unwrap(dataflowPath);
  return wrap(path->getRoot());
}

//===----------------------------------------------------------------------===//
// History
//===----------------------------------------------------------------------===//

size_t synthLongestPathHistoryGetSize(SynthLongestPathHistory history) {
  auto *vec = unwrap(history);
  return vec->size();
}

void synthLongestPathHistoryGetVal(SynthLongestPathHistory history, size_t index,
                                   SynthLongestPathObject *object,
                                   int64_t *delay, MlirStringRef *comment) {
  auto *vec = unwrap(history);

  if (index < vec->size()) {
    auto &point = (*vec)[index];
    *object = wrap(&point.object);
    *delay = point.delay;
    *comment = mlirStringRefCreate(point.comment.data(), point.comment.size());
  }
}

//===----------------------------------------------------------------------===//
// Object
//===----------------------------------------------------------------------===//

IgraphInstancePath
synthLongestPathObjectGetInstancePath(SynthLongestPathObject object) {
  auto *ptr = dyn_cast<Object *>(unwrap(object));
  if (ptr) {
    IgraphInstancePath result;
    result.ptr = const_cast<igraph::InstanceOpInterface *>(
        ptr->instancePath.getPath().data());
    result.size = ptr->instancePath.getPath().size();
    return result;
  }

  // This is output port so the instance path is empty.
  return {nullptr, 0};
}

MlirStringRef synthLongestPathObjectName(SynthLongestPathObject rawObject) {
  auto ptr = unwrap(rawObject);
  if (auto *object = dyn_cast<Object *>(ptr)) {
    auto name = object->getName();
    return mlirStringRefCreate(name.data(), name.size());
  }
  auto [module, resultNumber, _] = *dyn_cast<DataflowPath::OutputPort *>(ptr);
  auto name = module.getOutputName(resultNumber);
  return mlirStringRefCreate(name.data(), name.size());
}

size_t synthLongestPathObjectBitPos(SynthLongestPathObject rawObject) {
  auto ptr = unwrap(rawObject);
  if (auto *object = dyn_cast<Object *>(ptr))
    return object->bitPos;
  return std::get<2>(*dyn_cast<DataflowPath::OutputPort *>(ptr));
}

MlirValue synthLongestPathObjectGetValue(SynthLongestPathObject rawObject) {
  auto ptr = unwrap(rawObject);
  if (auto *object = dyn_cast<Object *>(ptr))
    return wrap(object->value);
  // Output ports don't have an associated value
  return {nullptr};
}
