//===------ CodeCompletion.h - Code Completion for ClangRepl -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes which performs code completion at the REPL.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_CODE_COMPLETION_H
#define LLVM_CLANG_INTERPRETER_CODE_COMPLETION_H
#include "clang/Sema/CodeCompleteConsumer.h"

namespace clang {

class ReplCompletionConsumer : public CodeCompleteConsumer {
public:
  ReplCompletionConsumer(std::vector<CodeCompletionResult> &Results);
  void ProcessCodeCompleteResults(class Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *InResults,
                                  unsigned NumResults) final;

  CodeCompletionAllocator &getAllocator() override { return *CCAllocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  std::shared_ptr<GlobalCodeCompletionAllocator> CCAllocator;
  CodeCompletionTUInfo CCTUInfo;
  std::vector<CodeCompletionResult> &Results;
};

std::vector<llvm::StringRef> ConvertToCodeCompleteStrings(
    const std::vector<clang::CodeCompletionResult> &Results);
} // namespace clang
#endif
