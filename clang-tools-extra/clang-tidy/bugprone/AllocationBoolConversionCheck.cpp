//===--- AllocationBoolConversionCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AllocationBoolConversionCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

AllocationBoolConversionCheck::AllocationBoolConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllocationFunctions(utils::options::parseStringList(
          Options.get("AllocationFunctions",
                      "malloc;calloc;realloc;strdup;fopen;fdopen;freopen;"
                      "opendir;fdopendir;popen;mmap;allocate"))) {}

void AllocationBoolConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllocationFunctions",
                utils::options::serializeStringList(AllocationFunctions));
}

void AllocationBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      castExpr(
          hasCastKind(CK_PointerToBoolean),
          hasSourceExpression(ignoringImplicit(anyOf(
              cxxNewExpr(),
              callExpr(callee(functionDecl(anyOf(hasAnyOverloadedOperatorName(
                                                     "new", "new[]"),
                                                 matchers::matchesAnyListedName(
                                                     AllocationFunctions)))
                                  .bind("func")))))))
          .bind("cast"),
      this);
}

void AllocationBoolConversionCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CastExpr>("cast");

  if (const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("func")) {
    diag(MatchedExpr->getExprLoc(),
         "result of the %0 call is being used as a boolean value, which "
         "may lead to unintended behavior or resource leaks")
        << Function;
  } else {
    diag(MatchedExpr->getExprLoc(),
         "result of the 'new' expression is being used as a boolean value, "
         "which may lead to unintended behavior or resource leaks");
  }
}

} // namespace clang::tidy::bugprone
