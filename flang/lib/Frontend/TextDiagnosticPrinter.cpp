//===--- TextDiagnosticPrinter.cpp - Diagnostic Printer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client prints out their diagnostic messages.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Frontend/TextDiagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <filesystem>
#include <string>
#include <vector>

using namespace Fortran::frontend;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &diagOs,
                                             clang::DiagnosticOptions *diags)
    : os(diagOs), diagOpts(diags) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {}

// For remarks only, print the remark option and pass name that was used to a
// raw_ostream.
static void printRemarkOption(llvm::raw_ostream &os,
                              clang::DiagnosticsEngine::Level level,
                              const clang::Diagnostic &info) {
  llvm::StringRef opt =
      clang::DiagnosticIDs::getWarningOptionForDiag(info.getID());
  if (!opt.empty()) {
    os << " [" << (level == clang::DiagnosticsEngine::Remark ? "-R" : "-W")
       << opt;
    llvm::StringRef optValue = info.getDiags()->getFlagValue();
    if (!optValue.empty())
      os << "=" << optValue;
    os << ']';
  }
}

void TextDiagnosticPrinter::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(level, info);

  // Render the diagnostic message into a temporary buffer eagerly. We'll use
  // this later as we print out the diagnostic to the terminal.
  llvm::SmallString<100> outStr;
  info.FormatDiagnostic(outStr);

  llvm::raw_svector_ostream diagMessageStream(outStr);
  printRemarkOption(diagMessageStream, level, info);

  if (!prefix.empty())
    os << prefix << ": ";

  // We only emit diagnostics in contexts that lack valid source locations.
  assert(!info.getLocation().isValid() &&
         "Diagnostics with valid source location are not supported");

  // split incoming string to get the absolute path and filename in the
  // case we are receiving optimization remarks from BackendRemarkConsumer
  std::string diagMsg = std::string(diagMessageStream.str());
  std::string delimiter = ";;";

  size_t pos = 0;
  std::vector<std::string> tokens;
  while ((pos = diagMsg.find(delimiter)) != std::string::npos) {
    tokens.push_back(diagMsg.substr(0, pos));
    diagMsg.erase(0, pos + delimiter.length());
  }

  // tokens will always be of size 2 in the case of optimization
  // remark message received, in this format;
  // [file location with line and column];;[path to file];;[the remark message]
  if (tokens.size() == 2) {
    // extract absolute path from the provided relative path
    std::filesystem::path absPath(tokens[1]);
    std::filesystem::path canonicalPath =
        std::filesystem::weakly_canonical(absPath);

    // we don't need the filename since we will append tokens[0]
    // which is the filename, line and column number
    canonicalPath.remove_filename();
    absPath = canonicalPath.make_preferred().string();

    // used for changing only the bold attribute
    if (diagOpts->ShowColors)
      os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);

    // print absolute path, file name, line and column
    os << absPath << tokens[0] << ": ";
  }

  Fortran::frontend::TextDiagnostic::printDiagnosticLevel(os, level,
                                                          diagOpts->ShowColors);
  Fortran::frontend::TextDiagnostic::printDiagnosticMessage(
      os,
      /*IsSupplemental=*/level == clang::DiagnosticsEngine::Note, diagMsg,
      diagOpts->ShowColors);

  os.flush();
}
