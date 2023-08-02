//===- MlirQueryMain.cpp - MLIR Query main ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the general framework of the MLIR query tool. It
// parses the command line arguments, parses the MLIR file and outputs the query
// results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-query/MlirQueryMain.h"
#include "mlir/Query/QueryParser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

//===----------------------------------------------------------------------===//
// Query Parser
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::mlirQueryMain(int argc, char **argv, MLIRContext &context,
                    const mlir::query::matcher::RegistryMaps &registryData) {

  // Override the default '-h' and use the default PrintHelpMessage() which
  // won't print options in categories.
  static llvm::cl::opt<bool> help("h", llvm::cl::desc("Alias for -help"),
                                  llvm::cl::Hidden);

  static llvm::cl::OptionCategory mlirQueryCategory("mlir-query options");

  static llvm::cl::list<std::string> commands(
      "c", llvm::cl::desc("Specify command to run"),
      llvm::cl::value_desc("command"), llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::cat(mlirQueryCategory));

  static llvm::cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      llvm::cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      llvm::cl::init(false)};

  static llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false));

  llvm::cl::HideUnrelatedOptions(mlirQueryCategory);

  llvm::InitLLVM y(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR test case query tool.\n");

  if (help) {
    llvm::cl::PrintHelpMessage();
    return success();
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  auto bufferId = sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());

  context.allowUnregisteredDialects(allowUnregisteredDialects);

  // Parse the input MLIR file.
  OwningOpRef<Operation *> opRef =
      parseSourceFileForTool(sourceMgr, &context, !noImplicitModule);
  if (!opRef)
    return failure();

  mlir::query::QuerySession QS(opRef.get(), sourceMgr, bufferId, registryData);
  if (!commands.empty()) {
    for (auto &command : commands) {
      mlir::query::QueryRef queryRef =
          mlir::query::QueryParser::parse(command, QS);
      if (!queryRef->run(llvm::outs(), QS))
        return failure();
    }
  } else {
    llvm::LineEditor LE("mlir-query");
    LE.setListCompleter([&QS](StringRef line, size_t pos) {
      return mlir::query::QueryParser::complete(line, pos, QS);
    });
    while (std::optional<std::string> line = LE.readLine()) {
      mlir::query::QueryRef queryRef =
          mlir::query::QueryParser::parse(*line, QS);
      queryRef->run(llvm::outs(), QS);
      llvm::outs().flush();
      if (QS.terminate)
        break;
    }
  }

  return success();
}
