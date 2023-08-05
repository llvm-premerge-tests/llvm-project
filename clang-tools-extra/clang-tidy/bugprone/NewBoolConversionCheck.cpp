//===--- NewBoolConversionCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NewBoolConversionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void NewBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      castExpr(hasCastKind(CK_PointerToBoolean),
               hasSourceExpression(ignoringImplicit(cxxNewExpr())))
          .bind("cast"),
      this);
}

void NewBoolConversionCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<CastExpr>("cast");

  diag(MatchedExpr->getExprLoc(),
       "result of the 'new' expression is being used as a boolean value, which "
       "may lead to unintended behavior or memory leaks");
}

} // namespace clang::tidy::bugprone
