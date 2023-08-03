//===--- TypeUtils.h - clang-tidy--------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPE_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPE_UTILS_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace clang {

class ASTContext;
class Decl;
class Preprocessor;

namespace tidy::utils::type {

struct ClassifiedToken {
  Token T;
  bool IsQualifier;
  bool IsSpecifier;
};

// Classify the qualifiers and specifier tokens of a declaration.
std::optional<SmallVector<ClassifiedToken, 8>>
classifyDeclTypeTokens(Preprocessor &PP, SourceLocation BeginLoc,
                       SourceLocation EndLoc, const ASTContext &Ctx);

std::optional<SmallVector<ClassifiedToken, 8>>
classifyDeclTypeTokens(Preprocessor &PP, const Decl &D, const ASTContext &Ctx);

} // namespace tidy::utils::type
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_TYPE_UTILS_H
