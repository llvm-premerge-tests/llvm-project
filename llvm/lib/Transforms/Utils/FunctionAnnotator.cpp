//===- FunctionAnnotator.cpp - Function Annotator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotates functions with the appropriate optimization level attribute for
// assistance in phase ordering.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionAnnotator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Annotations/Annotations.h"

using namespace llvm;

static cl::opt<std::string>
    OptLevelAttributeName("opt-level-attribute-name", cl::init(""), cl::Hidden,
                          cl::desc("Optimization attribute name"));

static cl::opt<std::string> CSVFilePath(
    "func-annotator-csv-file-path", cl::Hidden,
    cl::desc("CSV file containing function names and optimization level as "
             "attribute"));

static cl::opt<bool>
    IsHeaderRow("is-header-row", cl::init(true), cl::Hidden,
                cl::desc("True if the first row of the CSV file is a header "
                         "row, false otherwise"));

PreservedAnalyses FunctionAnnotator::run(Module &M, ModuleAnalysisManager &AM) {

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(CSVFilePath);
  if (!BufferOrError) {
    report_fatal_error("Cannot open CSV File");
  }

  StringRef Buffer = BufferOrError.get()->getBuffer();
  auto MemoryBuffer = MemoryBuffer::getMemBuffer(Buffer);
  line_iterator It(*MemoryBuffer);
  if (IsHeaderRow) {
    ++It;
  }
  while (!It.is_at_end()) {
    auto SplitPair = It->split(',');
    if (!SplitPair.second.empty()) {
      Function *Func = M.getFunction(SplitPair.first);
      if (Func) {
        if (Func->isDeclaration()) {
          ++It;
          continue;
        }
        Func->addFnAttr(OptLevelAttributeName, SplitPair.second);
      } else {
        outs() << "Function in CSV file at line " << It.line_number()
               << " does not exist\n";
        ++It;
        continue;
      }
    }
    ++It;
  }

  return PreservedAnalyses::all();
}
