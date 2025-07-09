//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_AIG_H
#define CIRCT_C_DIALECT_AIG_H

#include "mlir-c/IR.h"
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(AIG, aig);
MLIR_CAPI_EXPORTED void registerAIGPasses(void);

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

//===----------------------------------------------------------------------===//
// LongestPathAnalysis
//===----------------------------------------------------------------------===//

// Opaque handle to LongestPathAnalysis
struct AIGLongestPathObject {
  union {
    // LSB is reserved for a tag.
    // 0 indicates an object, 1 indicates an output port.
    void *object;

    // outputPort[0] is tag.
    // outputPort[1:31] is the output port number.
    // outputPort[32:63] is the bit position.
    uint64_t outputPort;
  } ptr;
};

// Opaque handle to LongestPathAnalysis
DEFINE_C_API_STRUCT(AIGLongestPathDataflowPath, void);

// Opaque handle to LongestPathAnalysis
DEFINE_C_API_STRUCT(AIGLongestPathHistory, void);

// Opaque handle to LongestPathAnalysis
DEFINE_C_API_STRUCT(AIGLongestPathAnalysis, void);

// Opaque handle to LongestPathCollection
DEFINE_C_API_STRUCT(AIGLongestPathCollection, void);

#undef DEFINE_C_API_STRUCT

// Create a LongestPathAnalysis for the given module
MLIR_CAPI_EXPORTED AIGLongestPathAnalysis
aigLongestPathAnalysisCreate(MlirOperation module, bool traceDebugPoints);

// Destroy a LongestPathAnalysis
MLIR_CAPI_EXPORTED void
aigLongestPathAnalysisDestroy(AIGLongestPathAnalysis analysis);

MLIR_CAPI_EXPORTED AIGLongestPathCollection aigLongestPathAnalysisGetAllPaths(
    AIGLongestPathAnalysis analysis, MlirStringRef moduleName,
    MlirStringRef fanoutFilter, MlirStringRef faninFilter, bool elaboratePaths);

//===----------------------------------------------------------------------===//
// LongestPathCollection
//===----------------------------------------------------------------------===//

// Check if the collection is valid
MLIR_CAPI_EXPORTED bool
aigLongestPathCollectionIsNull(AIGLongestPathCollection collection);

// Destroy a LongestPathCollection
MLIR_CAPI_EXPORTED void
aigLongestPathCollectionDestroy(AIGLongestPathCollection collection);

// Get the number of paths in the collection
MLIR_CAPI_EXPORTED size_t
aigLongestPathCollectionGetSize(AIGLongestPathCollection collection);

// Get a specific path from the collection as JSON
MLIR_CAPI_EXPORTED MlirStringRef aigLongestPathCollectionGetPathAsJson(
    AIGLongestPathCollection collection, int pathIndex);

MLIR_CAPI_EXPORTED AIGLongestPathDataflowPath aigLongestPathCollectionGetPath(
    AIGLongestPathCollection collection, int pathIndex);

MLIR_CAPI_EXPORTED int64_t
aigLongestPathDataflowPathGetDelay(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathObject
aigLongestPathDataflowPathGetFanIn(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathObject
aigLongestPathDataflowPathGetFanOut(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED AIGLongestPathHistory
aigLongestPathDataflowPathGetHistory(AIGLongestPathDataflowPath dataflowPath);

MLIR_CAPI_EXPORTED void aigLongestPathHistoryGet(AIGLongestPathHistory history,
                                                 size_t index,
                                                 AIGLongestPathObject *object,
                                                 int64_t *delay,
                                                 MlirAttribute *comment);

MLIR_CAPI_EXPORTED size_t
aigLongestPathHistoryGetSize(AIGLongestPathHistory history);

MLIR_CAPI_EXPORTED bool aigLongestPathCollectionDiff(
    AIGLongestPathCollection lhs, AIGLongestPathCollection rhs,
    AIGLongestPathCollection *uniqueLhs, AIGLongestPathCollection *uniqueRhs,
    AIGLongestPathCollection *differentLhs,
    AIGLongestPathCollection *differentRhs);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_AIG_H
