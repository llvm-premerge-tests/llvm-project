//===------ CodeCompletion.cpp - Code Completion for ClangRepl -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the classes which performs code completion at the REPL.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Sema/Sema.h"

namespace clang {

clang::CodeCompleteOptions getClangCompleteOpts() {
  clang::CodeCompleteOptions Opts;
  Opts.IncludeCodePatterns = true;
  Opts.IncludeMacros = true;
  Opts.IncludeGlobals = true;
  Opts.IncludeBriefComments = true;
  return Opts;
}

ReplCompletionConsumer::ReplCompletionConsumer(
    std::vector<CodeCompletionResult> &Results)
    : CodeCompleteConsumer(getClangCompleteOpts()),
      CCAllocator(std::make_shared<GlobalCodeCompletionAllocator>()),
      CCTUInfo(CCAllocator), Results(Results){};

void ReplCompletionConsumer::ProcessCodeCompleteResults(
    class Sema &S, CodeCompletionContext Context,
    CodeCompletionResult *InResults, unsigned NumResults) {
  for (unsigned I = 0; I < NumResults; ++I) {
    auto &Result = InResults[I];
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (Result.Declaration->getIdentifier()) {
        Results.push_back(Result);
      }
      break;
    default:
      break;
    case CodeCompletionResult::RK_Keyword:
      Results.push_back(Result);
      break;
    }
  }
}

std::vector<llvm::StringRef> ConvertToCodeCompleteStrings(
    const std::vector<clang::CodeCompletionResult> &Results) {
  std::vector<llvm::StringRef> CompletionStrings;
  for (auto Res : Results) {
    switch (Res.Kind) {
    case clang::CodeCompletionResult::RK_Declaration:
      if (auto *ID = Res.Declaration->getIdentifier()) {
        CompletionStrings.push_back(ID->getName());
      }
      break;
    case clang::CodeCompletionResult::RK_Keyword:
      CompletionStrings.push_back(Res.Keyword);
      break;
    default:
      break;
    }
  }
  return CompletionStrings;
}

} // namespace clang
