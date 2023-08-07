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
#include <iostream>
#include <string>
#include <vector>

using namespace Fortran::frontend;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &diagOs,
                                             clang::DiagnosticOptions *diags)
    : os(diagOs), diagOpts(diags) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {}

// Print any diagnostic option information to a raw_ostream.
static void printDiagnosticOptions(llvm::raw_ostream &os,
                                   clang::DiagnosticsEngine::Level level,
                                   const clang::Diagnostic &info,
                                   const clang::DiagnosticOptions &diagOpts) {
  bool started = false;
  if (diagOpts.ShowOptionNames) {
    llvm::StringRef opt =
        clang::DiagnosticIDs::getWarningOptionForDiag(info.getID());
    if (!opt.empty()) {
      os << (started ? "," : " [")
         << (level == clang::DiagnosticsEngine::Remark ? "-R" : "-W") << opt;
      llvm::StringRef optValue = info.getDiags()->getFlagValue();
      if (!optValue.empty())
        os << "=" << optValue;
      started = true;
    }
  }
  if (started)
    os << ']';
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
  printDiagnosticOptions(diagMessageStream, level, info, *diagOpts);

  if (!prefix.empty())
    os << prefix << ": ";

  // We only emit diagnostics in contexts that lack valid source locations.
  assert(!info.getLocation().isValid() &&
         "Diagnostics with valid source location are not supported");

  // split incoming string to get the absolute path and filename
  std::string s = std::string(diagMessageStream.str());
  std::string delimiter = ";";

  size_t pos = 0;
  std::vector<std::string> tokens;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    tokens.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());
  }

  // extract absolute path from the provided relative path
  std::string absPath = std::string(tokens[1]);
  std::filesystem::path path(absPath);
  std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(path);
  canonicalPath.remove_filename();
  absPath = canonicalPath.make_preferred().string();

  // used for changing only the bold attribute
  if (diagOpts->ShowColors)
    os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);

  // print absolute path, file name, line and column
  os << absPath << tokens[0] << ": ";

  Fortran::frontend::TextDiagnostic::printDiagnosticLevel(os, level,
                                                          diagOpts->ShowColors);
  Fortran::frontend::TextDiagnostic::printDiagnosticMessage(
      os,
      /*IsSupplemental=*/level == clang::DiagnosticsEngine::Note, s,
      diagOpts->ShowColors);

  os.flush();
}
