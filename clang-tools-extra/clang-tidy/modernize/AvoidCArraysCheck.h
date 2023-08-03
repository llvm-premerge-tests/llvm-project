//===--- AvoidCArraysCheck.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_AVOIDCARRAYSCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_AVOIDCARRAYSCHECK_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"

namespace clang::tidy::modernize {

/// Find C-style array types and recommend to use std::array<> / std::vector<>.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/avoid-c-arrays.html
class AvoidCArraysCheck : public ClangTidyCheck {
public:
  AvoidCArraysCheck(StringRef Name, ClangTidyContext *Context);
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus11;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

private:
  Preprocessor *PP = nullptr;
  utils::IncludeInserter IncludeInserter;
  bool replaceDecl(ArrayTypeLoc ATL, const VarDecl *Var, bool UseCTAD,
                   ASTContext &Context, std::vector<FixItHint> &FixIts);
  bool replaceArrayReferences(const VarDecl *Var, const FunctionDecl *Func,
                              ASTContext &Context,
                              std::vector<FixItHint> &FixIts);
  std::vector<FixItHint> replaceArray(ArrayTypeLoc ATL,
                                      const DeclStmt *VarDeclStmt,
                                      const VarDecl *Var, ASTContext &Context);
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_AVOIDCARRAYSCHECK_H
