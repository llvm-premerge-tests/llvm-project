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
#include <array>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

static const std::array<llvm::StringRef, 33> PointerReturningAllocators{
    {"malloc",
     "calloc",
     "realloc",
     "aligned_alloc",
     "allocate",
     "fopen",
     "fdopen",
     "freopen",
     "popen",
     "tmpfile",
     "::opendir",
     "::fdopendir",
     "::mmap",
     "::reallocf",
     "::strdup",
     "::wcsdup",
     "::strndup",
     "::realpath",
     "::tempnam",
     "::canonicalize_file_name",
     "::dbopen",
     "::fmemopen",
     "::open_memstream",
     "::open_wmemstream",
     "::get_current_dir_name",
     "memalloc",
     "memcalloc",
     "memrealloc",
     "::mpool_open",
     "::posix_memalign",
     "::memalign",
     "::valloc",
     "::pvalloc"}};

static const std::array<llvm::StringRef, 16> IntegerReturningAllocators{
    {"::open", "::openat", "::creat", "::dup", "::dup2", "::dup3", "::socket",
     "::accept", "::pipe", "::pipe2", "::mkfifo", "::mkfifoat", "::mkstemp",
     "::mkostemp", "::mkstemps", "::mkostemps"}};

AllocationBoolConversionCheck::AllocationBoolConversionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      PointerReturningAllocatorsConfig(utils::options::parseStringList(
          Options.get("PointerReturningAllocators", ""))),
      IntegerReturningAllocatorsConfig(utils::options::parseStringList(
          Options.get("IntegerReturningAllocators", ""))) {}

void AllocationBoolConversionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(
      Opts, "PointerReturningAllocators",
      utils::options::serializeStringList(PointerReturningAllocatorsConfig));
  Options.store(
      Opts, "IntegerReturningAllocators",
      utils::options::serializeStringList(IntegerReturningAllocatorsConfig));
}

void AllocationBoolConversionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      castExpr(hasCastKind(CK_PointerToBoolean),
               hasSourceExpression(ignoringImplicit(anyOf(
                   cxxNewExpr(),
                   callExpr(callee(
                       functionDecl(
                           anyOf(hasAnyOverloadedOperatorName("new", "new[]"),
                                 matchers::matchesAnyListedName(
                                     PointerReturningAllocators),
                                 matchers::matchesAnyListedName(
                                     PointerReturningAllocatorsConfig)))
                           .bind("func")))))))
          .bind("cast"),
      this);

  Finder->addMatcher(
      castExpr(hasCastKind(CK_IntegralToBoolean),
               hasSourceExpression(ignoringImplicit(callExpr(callee(
                   functionDecl(anyOf(matchers::matchesAnyListedName(
                                          IntegerReturningAllocators),
                                      matchers::matchesAnyListedName(
                                          IntegerReturningAllocatorsConfig)))
                       .bind("func"))))))
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
