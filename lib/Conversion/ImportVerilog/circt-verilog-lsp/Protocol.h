//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs for LSP commands that are specific to the PDLL
// server.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_PROTOCOL_H_
#define LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_PROTOCOL_H_

#include "mlir/Tools/lsp-server-support/Protocol.h"

namespace circt {
namespace lsp {
//===----------------------------------------------------------------------===//
// PDLLViewOutputParams
//===----------------------------------------------------------------------===//

/// The type of output to view from PDLL.
enum class VerilogViewOutputKind {
  AST,
  MLIR,
  CPP,
};

/// Represents the parameters used when viewing the output of a PDLL file.
struct VerilogViewOutputParams {
  /// The URI of the document to view the output of.
  mlir::lsp::URIForFile uri;

  /// The kind of output to generate.
  VerilogViewOutputKind kind;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, VerilogViewOutputKind &result,
              llvm::json::Path path);
bool fromJSON(const llvm::json::Value &value, VerilogViewOutputParams &result,
              llvm::json::Path path);

//===----------------------------------------------------------------------===//
// PDLLViewOutputResult
//===----------------------------------------------------------------------===//

/// Represents the result of viewing the output of a Verilog file.
struct VerilogViewOutputResult {
  /// The string representation of the output.
  std::string output;
};

/// Add support for JSON serialization.
llvm::json::Value toJSON(const VerilogViewOutputResult &value);

//===----------------------------------------------------------------------===//
// VerilogObjectPathInlayHintsParams
//===----------------------------------------------------------------------===//

/// Represents the parameters used when viewing the output of a PDLL file.
struct VerilogObjectPathAndValue {
  /// The path to the value.
  std::string path;
  /// The value.
  std::string value;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value, VerilogObjectPathAndValue &result,
              llvm::json::Path path);

struct VerilogObjectPathInlayHintsParams {
  /// The URI of the document to view the output of.
  std::vector<VerilogObjectPathAndValue> values;
};

/// Add support for JSON serialization.
bool fromJSON(const llvm::json::Value &value,
              VerilogObjectPathInlayHintsParams &result, llvm::json::Path path);

// Consider using this directly from MLIR file.
struct VerilogInstanceHierarchy {
  std::string module_name;
  std::string instance_name;
  std::vector<std::unique_ptr<VerilogInstanceHierarchy>> instances;
  VerilogInstanceHierarchy *parent = nullptr;
};

} // namespace lsp
} // namespace circt

#endif