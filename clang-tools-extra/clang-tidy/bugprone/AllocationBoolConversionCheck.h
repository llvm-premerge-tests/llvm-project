//===--- AllocationBoolConversionCheck.h - clang-tidy -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ALLOCATIONBOOLCONVERSIONCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ALLOCATIONBOOLCONVERSIONCHECK_H

#include "../ClangTidyCheck.h"
#include <vector>

namespace clang::tidy::bugprone {

/// Detects cases where the result of a resource allocation is used as a
/// ``bool``.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/bugprone/allocation-bool-conversion.html
class AllocationBoolConversionCheck : public ClangTidyCheck {
public:
  AllocationBoolConversionCheck(StringRef Name, ClangTidyContext *Context);
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  std::vector<llvm::StringRef> AllocationFunctions;
};

} // namespace clang::tidy::bugprone

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_BUGPRONE_ALLOCATIONBOOLCONVERSIONCHECK_H
