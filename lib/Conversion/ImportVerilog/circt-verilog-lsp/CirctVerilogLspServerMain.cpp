//===- .cpp - MLIR PDLL Language Server main ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-verilog-lsp/CirctVerilogLspServerMain.h"
#include "LSPServer.h"
#include "VerilogServer.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Program.h"

using namespace mlir;
using namespace mlir::lsp;
// using namespace circt::lsp;

llvm::LogicalResult circt::lsp::CirctVerilogLspServerMain(
    const circt::lsp::VerilogServerOptions &options,
    mlir::lsp::JSONTransport &transport) {
  // // LSP options.
  // llvm::cl::list<std::string> extraIncludeDirs(
  //     "verilog-include-dir", llvm::cl::desc("Extra directory of Verilog files"),
  //     llvm::cl::value_desc("directory"), llvm::cl::Prefix);
  // llvm::cl::list<std::string> sourceLocationIncludeDirs(
  //     "source-location-include-dir",
  //     llvm::cl::desc("Root directory of file source locations"),
  //     llvm::cl::value_desc("directory"), llvm::cl::Prefix);
  // llvm::cl::opt<Logger::Level> logLevel{
  //     "log",
  //     llvm::cl::desc("Verbosity of log messages written to stderr"),
  //     llvm::cl::values(
  //         clEnumValN(Logger::Level::Error, "error", "Error messages only"),
  //         clEnumValN(Logger::Level::Info, "info",
  //                    "High level execution tracing"),
  //         clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
  //     llvm::cl::init(Logger::Level::Info),
  // };
  // llvm::cl::list<std::string> inlayHintFiles(
  //     "inlayHintFiles", llvm::cl::desc("Static files to display inlay hints"),
  //     llvm::cl::value_desc("directory"), llvm::cl::Prefix);

  // // Testing.
  // llvm::cl::opt<bool> prettyPrint{
  //     "pretty",
  //     llvm::cl::desc("Pretty-print JSON output"),
  //     llvm::cl::init(false),
  // };
  // llvm::cl::opt<bool> litTest{
  //     "lit-test",
  //     llvm::cl::desc(
  //         "Abbreviation for -input-style=delimited -pretty -log=verbose. "
  //         "Intended to simplify lit tests"),
  //     llvm::cl::init(false),
  // };

  circt::lsp::VerilogServer server(options);
  return circt::lsp::runVerilogLSPServer(server, transport);
}
