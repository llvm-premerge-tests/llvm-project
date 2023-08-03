//===--- TypeUtils.cpp - clang-tidy----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>

namespace clang::tidy::utils::type {

static bool isCvr(Token T) {
  return T.isOneOf(tok::kw_const, tok::kw_volatile, tok::kw_restrict);
}

static bool isSpecifier(Token T) {
  return T.isOneOf(tok::kw_constexpr, tok::kw_inline, tok::kw_extern,
                   tok::kw_static, tok::kw_friend, tok::kw_virtual,
                   tok::kw_thread_local, tok::kw_register);
}

static std::optional<ClassifiedToken> classifyToken(Preprocessor &PP,
                                                    Token Tok) {
  ClassifiedToken CT;
  CT.T = Tok;
  CT.IsQualifier = true;
  CT.IsSpecifier = true;
  bool ContainsQualifiers = false;
  bool ContainsSpecifiers = false;
  bool ContainsSomethingElse = false;

  Token End;
  End.startToken();
  End.setKind(tok::eof);
  SmallVector<Token, 2> Stream{Tok, End};

  // FIXME: do not report these token to Preprocessor.TokenWatcher.
  PP.EnterTokenStream(Stream, false, /*IsReinject=*/false);
  while (true) {
    Token T;
    PP.Lex(T);
    if (T.is(tok::eof))
      break;

    const bool Qual = isCvr(T);
    const bool Spec = isSpecifier(T);
    CT.IsQualifier &= Qual;
    CT.IsSpecifier &= Spec;
    ContainsQualifiers |= Qual;
    ContainsSpecifiers |= Spec;
    ContainsSomethingElse |= !Qual && !Spec;
  }

  // If the Token/Macro contains more than one type of tokens, we would need
  // to split the macro in order to move parts to the trailing return type.
  if (ContainsQualifiers + ContainsSpecifiers + ContainsSomethingElse > 1)
    return std::nullopt;

  return CT;
}

static SourceLocation expandIfMacroId(SourceLocation Loc,
                                      const SourceManager &SM) {
  if (Loc.isMacroID())
    Loc = expandIfMacroId(SM.getImmediateExpansionRange(Loc).getBegin(), SM);
  assert(!Loc.isMacroID() &&
         "SourceLocation must not be a macro ID after recursive expansion");
  return Loc;
}

std::optional<SmallVector<ClassifiedToken, 8>>
classifyDeclTypeTokens(Preprocessor &PP, SourceLocation BeginLoc,
                       SourceLocation EndLoc, const ASTContext &Ctx) {
  const SourceManager &SM = Ctx.getSourceManager();
  const LangOptions &LangOpts = Ctx.getLangOpts();
  BeginLoc = expandIfMacroId(BeginLoc, SM);
  EndLoc = expandIfMacroId(EndLoc, SM);

  // Create tokens for everything before the name of the function.
  std::pair<FileID, unsigned> Loc = SM.getDecomposedLoc(BeginLoc);
  StringRef File = SM.getBufferData(Loc.first);
  const char *TokenBegin = File.data() + Loc.second;
  Lexer Lexer(SM.getLocForStartOfFile(Loc.first), LangOpts, File.begin(),
              TokenBegin, File.end());
  Token T;
  SmallVector<ClassifiedToken, 8> ClassifiedTokens;
  while (!Lexer.LexFromRawLexer(T) &&
         SM.isBeforeInTranslationUnit(T.getLocation(), EndLoc)) {
    if (T.is(tok::raw_identifier)) {
      IdentifierInfo &Info = Ctx.Idents.get(
          StringRef(SM.getCharacterData(T.getLocation()), T.getLength()));

      if (Info.hasMacroDefinition()) {
        const MacroInfo *MI = PP.getMacroInfo(&Info);
        if (!MI || MI->isFunctionLike()) {
          // Cannot handle function style macros.
          return std::nullopt;
        }
      }

      T.setIdentifierInfo(&Info);
      T.setKind(Info.getTokenID());
    }

    if (std::optional<ClassifiedToken> CT = classifyToken(PP, T))
      ClassifiedTokens.push_back(*CT);
    else
      return std::nullopt;
  }

  return ClassifiedTokens;
}

std::optional<SmallVector<ClassifiedToken, 8>>
classifyDeclTypeTokens(Preprocessor &PP, const Decl &D, const ASTContext &Ctx) {
  return classifyDeclTypeTokens(PP, D.getBeginLoc(), D.getLocation(), Ctx);
}

} // namespace clang::tidy::utils::type
