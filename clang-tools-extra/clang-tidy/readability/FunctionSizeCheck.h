//===--- FunctionSizeCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::readability {

/// Checks for large functions based on various metrics.
///
/// These options are supported:
///
///   * `LineThreshold` - flag functions exceeding this number of lines. This
///     parameter is disabled by default.
///   * `StatementThreshold` - flag functions exceeding this number of
///     statements. This may differ significantly from the number of lines for
///     macro-heavy code. The default is `800`.
///   * `BranchThreshold` - flag functions exceeding this number of control
///     statements. This parameter is disabled by default.
///   * `ParameterThreshold` - flag functions having a high number of
///     parameters. This parameter is disabled by default.
///   * `NestingThreshold` - flag compound statements which create next nesting
///     level after `NestingThreshold`. This may differ significantly from the
///     expected value for macro-heavy code. This parameter is disabled by
///     default.
///   * `VariableThreshold` - flag functions having a high number of variable
///     declarations. This parameter is disabled by default.
class FunctionSizeCheck : public ClangTidyCheck {
public:
  FunctionSizeCheck(StringRef Name, ClangTidyContext *Context);

  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

private:
  const std::optional<unsigned> LineThreshold;
  const std::optional<unsigned> StatementThreshold;
  const std::optional<unsigned> BranchThreshold;
  const std::optional<unsigned> ParameterThreshold;
  const std::optional<unsigned> NestingThreshold;
  const std::optional<unsigned> VariableThreshold;

  static constexpr std::optional<unsigned> DefaultLineThreshold = std::nullopt;
  static constexpr std::optional<unsigned> DefaultStatementThreshold = 800U;
  static constexpr std::optional<unsigned> DefaultBranchThreshold =
      std::nullopt;
  static constexpr std::optional<unsigned> DefaultParameterThreshold =
      std::nullopt;
  static constexpr std::optional<unsigned> DefaultNestingThreshold =
      std::nullopt;
  static constexpr std::optional<unsigned> DefaultVariableThreshold =
      std::nullopt;
};

} // namespace clang::tidy::readability

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_READABILITY_FUNCTIONSIZECHECK_H
