//===--- AvoidCArraysCheck.cpp - clang-tidy -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidCArraysCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/TypeUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>
#include <tuple>

using namespace clang::ast_matchers;

namespace {

AST_MATCHER(clang::TypeLoc, hasValidBeginLoc) {
  return Node.getBeginLoc().isValid();
}

AST_MATCHER_P(clang::TypeLoc, hasType,
              clang::ast_matchers::internal::Matcher<clang::Type>,
              InnerMatcher) {
  const clang::Type *TypeNode = Node.getTypePtr();
  return TypeNode != nullptr &&
         InnerMatcher.matches(*TypeNode, Finder, Builder);
}

AST_MATCHER(clang::RecordDecl, isExternCContext) {
  return Node.isExternCContext();
}

AST_MATCHER(clang::ParmVarDecl, isArgvOfMain) {
  const clang::DeclContext *DC = Node.getDeclContext();
  const auto *FD = llvm::dyn_cast<clang::FunctionDecl>(DC);
  return FD ? FD->isMain() : false;
}

} // namespace

namespace clang::tidy::modernize {

using utils::type::ClassifiedToken;

void AvoidCArraysCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      typeLoc(hasValidBeginLoc(), hasType(arrayType()),
              optionally(hasParent(varDecl(hasParent(declStmt().bind("decl")),
                                           hasAncestor(functionDecl()))
                                       .bind("var"))),
              unless(anyOf(hasParent(parmVarDecl(isArgvOfMain())),
                           hasParent(varDecl(isExternC())),
                           hasParent(fieldDecl(
                               hasParent(recordDecl(isExternCContext())))),
                           hasAncestor(functionDecl(isExternC())))))
          .bind("typeloc"),
      this);
}

SourceLocation findEndOfToken(SourceLocation Loc, const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  SourceLocation Orig = Loc;
  Loc = Loc.getLocWithOffset(1);
  while (Loc.isValid()) {
    SourceLocation PrevTokenLoc =
        utils::lexer::findPreviousTokenStart(Loc, SM, LangOpts);
    if (PrevTokenLoc != Orig)
      return Loc.getLocWithOffset(-1);
    Loc = Loc.getLocWithOffset(1);
  }
  return SourceLocation{};
}

SourceLocation getPreviousTokenLoc(SourceLocation Location,
                                   const SourceManager &SM,
                                   const LangOptions &LangOpts,
                                   bool SkipComments = false) {
  Token Token;
  Token.setKind(tok::unknown);

  Location = Location.getLocWithOffset(-1);
  if (Location.isInvalid())
    return Location;

  SourceLocation StartOfFile = SM.getLocForStartOfFile(SM.getFileID(Location));
  while (Location != StartOfFile) {
    Location = Lexer::GetBeginningOfToken(Location, SM, LangOpts);
    if (!Lexer::getRawToken(Location, Token, SM, LangOpts) &&
        (!SkipComments || !Token.is(tok::comment)))
      break;
    Location = Location.getLocWithOffset(-1);
  }
  return Location;
}

SourceLocation getVarLocForReplacement(const VarDecl *Var,
                                       const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  SourceLocation EndLoc = getPreviousTokenLoc(Var->getLocation(), SM, LangOpts);
  if (!EndLoc.isValid())
    return SourceLocation{};
  EndLoc = findEndOfToken(EndLoc, Context);
  if (!EndLoc.isValid())
    return SourceLocation{};
  CharSourceRange Range{SourceRange{EndLoc, EndLoc.getLocWithOffset(1)}, false};
  StringRef Text = Lexer::getSourceText(Range, SM, LangOpts);
  if (Text.empty())
    return SourceLocation{};
  if (std::isspace(Text[0]))
    return EndLoc.getLocWithOffset(1);
  return EndLoc;
}

// Collects the qualifiers, specifiers, and type tokens and combines them
// into source text strings suitable for std::array<>/std::vector<>
// replacement. Specifiers are placed outside of the array/vector template
// type, and the qualifiers and type tokens themselves are placed within
// the array/vector template type.
//
// Any text between the final non-specifer token and the following specifier
// token is moved to appear after the ">" in the eventual
// std::array<>/std::vector<> declaration.
//
// The logic works in two passes: first, classifying the tokens and collecting
// the source text associated with each token including any whitespace/comments
// that should remain "attached" to the token. Second, the tokens and text
// are parsed using a "simple" algorithm to determine whether qualifiers are
// associated with the array, or the array element. In particular, handling
// pointers requires extra care to distinguish a qualifier on the pointer type
// vs qualifier on the type being pointed to. Since QualType/TypeLoc do not
// store per qualifier source location information, this bespoke logic cannot
// handle every possible case (see tests marked with FIXMEs), but it gets
// the most common cases.
std::optional<std::tuple<std::string, std::string, std::string>>
extractTypeComponents(Preprocessor *PP, TypeLoc TL, const VarDecl *Var,
                      const ASTContext &Context) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  SourceLocation EndLoc = getVarLocForReplacement(Var, Context);
  if (!EndLoc.isValid())
    return std::nullopt;
  std::optional<SmallVector<ClassifiedToken, 8>> MaybeTokens =
      utils::type::classifyDeclTypeTokens(*PP, Var->getBeginLoc(), EndLoc,
                                          Context);
  if (!MaybeTokens || MaybeTokens->size() == 0)
    return std::nullopt;

  unsigned LastTypeIndex = -1;
  for (auto [Index, Token] : llvm::enumerate(*MaybeTokens)) {
    if (!Token.IsSpecifier && !Token.IsQualifier)
      LastTypeIndex = Index;
  }

  struct TokenAndText {
    ClassifiedToken T;
    std::string Text;

    TokenAndText(ClassifiedToken T, StringRef Text) : T(T), Text(Text) {}
  };

  std::string Specifiers;
  std::string QualType;
  std::string BaseType;
  std::string Suffix;

  SmallVector<TokenAndText, 8> Tokens;
  for (auto [Index, Current] : llvm::enumerate(*MaybeTokens)) {
    if (Index == MaybeTokens->size() - 1) {
      if (Index == LastTypeIndex) {
        CharSourceRange CSR{SourceRange{Current.T.getLocation()}, true};
        Tokens.emplace_back(Current, Lexer::getSourceText(CSR, SM, LangOpts));

        SourceLocation SuffixStart =
            findEndOfToken(Current.T.getLocation(), Context);
        Suffix = Lexer::getSourceText(
            CharSourceRange{SourceRange{SuffixStart, EndLoc}, false}, SM,
            LangOpts);
      } else {
        CharSourceRange CSR{SourceRange{Current.T.getLocation(), EndLoc},
                            false};
        Tokens.emplace_back(Current, Lexer::getSourceText(CSR, SM, LangOpts));
      }
    } else {
      const ClassifiedToken &Next = (*MaybeTokens)[Index + 1];
      if (Index == LastTypeIndex) {
        Tokens.emplace_back(
            Current,
            Lexer::getSourceText(
                CharSourceRange{SourceRange{Current.T.getLocation()}, true}, SM,
                LangOpts));

        SourceLocation SuffixStart =
            findEndOfToken(Current.T.getLocation(), Context);
        Suffix = Lexer::getSourceText(
            CharSourceRange{SourceRange{SuffixStart, Next.T.getLocation()},
                            false},
            SM, LangOpts);
      } else {
        Tokens.emplace_back(
            Current, Lexer::getSourceText(
                         CharSourceRange{SourceRange{Current.T.getLocation(),
                                                     Next.T.getLocation()},
                                         false},
                         SM, LangOpts));
      }
    }
  }

  // Track whether we've seen a '*' token while scanning right to left.
  bool ConsumedStar = false;
  std::optional<SourceLocation> StarLoc;
  if (PointerTypeLoc PTL = TL.getUnqualifiedLoc().getAs<PointerTypeLoc>())
    StarLoc.emplace(PTL.getStarLoc());

  // Track if we correctly found the CV qualifiers of the array. After
  // processing the tokens, validate our assumptions and bail out if
  // any assumption is not correct.
  bool SawConst = false;
  bool SawVolatile = false;

  // TypeLoc'c SourceLocation starts at the first non-qualifier, so any
  // associated type qualifier is not included. However, type qualifiers
  // may be embeded in the type ('unsigned const int').
  auto WithinTypeRange = [&, UnqualifiedTL = TL.getUnqualifiedLoc()](
                             const ClassifiedToken &Token) -> bool {
    if (!Token.IsQualifier)
      return false;

    FullSourceLoc TokLoc{Token.T.getLocation(), SM};
    return FullSourceLoc(UnqualifiedTL.getBeginLoc(), SM)
               .isBeforeInTranslationUnitThan(TokLoc) &&
           TokLoc.isBeforeInTranslationUnitThan(UnqualifiedTL.getEndLoc());
  };
  auto IsSpec = [&](const ClassifiedToken &Token) -> bool {
    if (WithinTypeRange(Token))
      return false;

    if (ConsumedStar)
      return Token.IsSpecifier;
    else
      return Token.IsSpecifier || Token.IsQualifier;
  };
  auto AppendTypeToken = [&](const ClassifiedToken &Token, StringRef Text) {
    if (StarLoc.has_value() && Token.T.getLocation() == *StarLoc)
      ConsumedStar = true;

    if (!WithinTypeRange(Token) && !ConsumedStar) {
      if (Token.T.is(tok::kw_const))
        SawConst = true;
      if (Token.T.is(tok::kw_volatile))
        SawVolatile = true;
    }

    if (IsSpec(Token)) {
      Specifiers = (Text + Specifiers).str();
    } else {
      if (ConsumedStar)
        QualType = (Text + QualType).str();
      else
        BaseType = (Text + BaseType).str();
    }
  };
  for (const auto &T : llvm::reverse(Tokens))
    AppendTypeToken(T.T, T.Text);

  if (ConsumedStar)
    Specifiers += BaseType;
  else {
    assert(QualType.size() == 0);
    QualType = BaseType;
  }

  if (TL.getType().isLocalVolatileQualified() ^ SawVolatile)
    return {};
  if (TL.getType().isLocalConstQualified() ^ SawConst)
    return {};

  return std::make_tuple(std::move(Specifiers), std::move(QualType),
                         std::move(Suffix));
}

bool AvoidCArraysCheck::replaceDecl(ArrayTypeLoc ATL, const VarDecl *Var,
                                    bool UseCTAD, ASTContext &Context,
                                    std::vector<FixItHint> &FixIts) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  std::optional<std::tuple<std::string, std::string, std::string>>
      TypeComponents =
          extractTypeComponents(PP, ATL.getElementLoc(), Var, Context);
  if (!TypeComponents)
    return {};
  auto [Specifiers, QualType, Suffix] = *TypeComponents;

  if (UseCTAD && Suffix.empty())
    Suffix = " ";

  std::string Replacement;
  const bool IsVLA = ATL.getTypePtr()->isVariableArrayType();
  const bool IsCharElem =
      ATL.getElementLoc().getTypePtr()->isAnyCharacterType();
  if (UseCTAD) {
    // FIXME - support vector
    if (IsVLA)
      return false;
    else
      Replacement += Specifiers + "std::array" + Suffix;
  } else {
    // FIXME - support vector
    if (IsVLA)
      return false;
    else
      Replacement += Specifiers + "std::array<" + QualType + ", ";
    if (ATL.getSizeExpr()) {
      Replacement += Lexer::getSourceText(
          CharSourceRange::getCharRange(
              ATL.getLBracketLoc().getLocWithOffset(1), ATL.getRBracketLoc()),
          SM, LangOpts);
    } else {
      if (!Var->hasInit())
        return false;

      const Expr *InitExpr = Var->getInit();
      if (const auto *EWC = dyn_cast<ExprWithCleanups>(InitExpr))
        InitExpr = EWC->getSubExpr();

      if (const auto *ILE = dyn_cast<InitListExpr>(InitExpr)) {
        if (IsCharElem) {
          if (ILE->getNumInits() != 1)
            return false;
          const Expr *FirstInitExpr = ILE->getInit(0);
          if (const auto *SE = dyn_cast<StringLiteral>(FirstInitExpr))
            Replacement += std::to_string(SE->getLength() + 1);
          else
            return false;
        } else {
          Replacement += std::to_string(ILE->getNumInits());
        }
      } else if (const auto *SE = dyn_cast<StringLiteral>(InitExpr))
        Replacement += std::to_string(SE->getLength() + 1);
      else
        return false;
    }
    Replacement += ">" + Suffix;
  }

  SourceLocation EndLoc = getVarLocForReplacement(Var, Context);
  CharSourceRange ReplacementRange =
      CharSourceRange::getCharRange(Var->getBeginLoc(), EndLoc);

  FixIts.push_back(FixItHint::CreateReplacement(ReplacementRange, Replacement));
  FixIts.push_back(FixItHint::CreateRemoval(ATL.getBracketsRange()));
  return true;
}

bool AvoidCArraysCheck::replaceArrayReferences(const VarDecl *Var,
                                               const FunctionDecl *Func,
                                               ASTContext &Context,
                                               std::vector<FixItHint> &FixIts) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  auto NonParensExpr = [](auto &&...Matchers) {
    return expr(std::forward<decltype(Matchers)>(Matchers)...,
                unless(parenExpr()));
  };
  SmallVector<BoundNodes, 4> Matches = match(
      findAll(declRefExpr(
                  to(equalsNode(Var)), unless(hasAncestor(typeAliasDecl())),
                  optionally(hasParent(varDecl(hasParent(declStmt(
                      hasParent(cxxForRangeStmt().bind("range-for"))))))),
                  optionally(hasAncestor(
                      NonParensExpr(
                          optionally(unaryExprOrTypeTraitExpr().bind("sizeof")),
                          optionally(
                              hasAncestor(NonParensExpr().bind("parent2Expr"))))
                          .bind("parent1"))))
                  .bind("ref")),
      *Func->getBody(), Context);
  for (const BoundNodes &Match : Matches) {
    const DeclRefExpr *Ref = Match.getNodeAs<DeclRefExpr>("ref");
    if (Match.getNodeAs<CXXForRangeStmt>("range-for"))
      continue;
    if (Match.getNodeAs<UnaryExprOrTypeTraitExpr>("sizeof"))
      continue;

    const Expr *Parent1 = Match.getNodeAs<Expr>("parent1");
    const Expr *Parent2Expr = Match.getNodeAs<Expr>("parent2Expr");
    if (!Parent1)
      return false;

    if (const auto *Cast = dyn_cast<ImplicitCastExpr>(Parent1)) {
      if (Cast->getCastKind() != CK_ArrayToPointerDecay)
        return false;
      if (const auto *Subscript =
              dyn_cast_or_null<ArraySubscriptExpr>(Parent2Expr)) {
        if (Subscript->getLHS() == Parent1) {
          // Nothing to change in this case, e.g., 'arr[1]'
          continue;
        } else {
          // Swap order of subscript, e.g., '1[arr]'
          const bool NotTokenRange = true;
          CharSourceRange NameRange{
              SourceRange{Subscript->getBase()->getBeginLoc(),
                          Subscript->getBase()->getEndLoc()},
              NotTokenRange};
          StringRef Name = Lexer::getSourceText(NameRange, SM, LangOpts);
          CharSourceRange IdxRange{
              SourceRange{Subscript->getIdx()->getBeginLoc(),
                          Subscript->getIdx()->getEndLoc()},
              NotTokenRange};
          StringRef Idx = Lexer::getSourceText(IdxRange, SM, LangOpts);
          FixIts.push_back(FixItHint::CreateReplacement(NameRange, Idx));
          FixIts.push_back(FixItHint::CreateReplacement(IdxRange, Name));
        }
      } else {
        const bool IsTokenRange = true;
        if (Ref->getBeginLoc().isMacroID())
          return false;

        CharSourceRange NameRange{
            SourceRange{Ref->getBeginLoc(), Ref->getEndLoc()}, IsTokenRange};
        StringRef Name = Lexer::getSourceText(NameRange, SM, LangOpts);
        FixIts.push_back(
            FixItHint::CreateReplacement(NameRange, (Name + ".begin()").str()));
        continue;
      }
    } else
      return false;
  }

  return true;
}

std::vector<FixItHint>
AvoidCArraysCheck::replaceArray(ArrayTypeLoc ATL, const DeclStmt *VarDeclStmt,
                                const VarDecl *Var, ASTContext &Context) {
  const FunctionDecl *Func = dyn_cast<FunctionDecl>(Var->getDeclContext());
  if (!Func || !Func->getBody())
    return {};

  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  const Type *ElemType = ATL.getElementLoc().getTypePtr();
  if (ElemType->isArrayType() || ElemType->isFunctionPointerType() ||
      ElemType->isMemberFunctionPointerType())
    return {};

  std::vector<FixItHint> FixIts;
  if (!replaceArrayReferences(Var, Func, Context, FixIts))
    return {};

  if (!VarDeclStmt->isSingleDecl())
    return {};

  const Expr *InitExpr = Var->hasInit() ? Var->getInit() : nullptr;
  if (const auto *EWC = dyn_cast_or_null<ExprWithCleanups>(InitExpr))
    InitExpr = EWC->getSubExpr();

  // Determine if we can use a CTAD declaration
  const bool IsCharElem =
      ATL.getElementLoc().getTypePtr()->isAnyCharacterType();
  bool UseCTAD = false;
  if (getLangOpts().CPlusPlus17 && InitExpr && !ATL.getSizeExpr() &&
      !IsCharElem) {
    if (const auto *ILE = dyn_cast<InitListExpr>(Var->getInit())) {
      if (ILE->getNumInits() > 0) {
        UseCTAD = llvm::all_of(
            ILE->inits(),
            [ElemType =
                 ATL.getElementLoc().getType().getTypePtr()](const Expr *E) {
              const Expr *SpelledExpr = E->IgnoreUnlessSpelledInSource();
              if (dyn_cast<InitListExpr>(SpelledExpr))
                return false;
              const auto *ConstructExpr =
                  dyn_cast<CXXConstructExpr>(SpelledExpr);
              if (ConstructExpr &&
                  !dyn_cast<CXXTemporaryObjectExpr>(SpelledExpr) &&
                  ConstructExpr->isListInitialization())
                return false;
              return SpelledExpr->getType().getTypePtr() == ElemType;
            });
      }
    }
  }

  // Add FixIt {} around initializer
  if (InitExpr && !UseCTAD) {
    SourceRange InitRange;
    if (const auto *SE = dyn_cast<StringLiteral>(InitExpr)) {
      std::optional<Token> NextToken =
          Lexer::findNextToken(SE->getEndLoc(), SM, LangOpts);
      if (!NextToken)
        return {};
      InitRange = SourceRange{SE->getBeginLoc(), NextToken->getLocation()};
    } else if (const auto *ILE = dyn_cast<InitListExpr>(InitExpr)) {
      std::optional<Token> NextToken =
          Lexer::findNextToken(ILE->getEndLoc(), SM, LangOpts);
      if (!NextToken)
        return {};
      InitRange = SourceRange{ILE->getBeginLoc(), NextToken->getLocation()};
    }

    if (InitRange.isValid()) {
      if (utils::lexer::rangeContainsExpansionsOrDirectives(InitRange, SM,
                                                            LangOpts))
        return {};
      FixIts.push_back(FixItHint::CreateInsertion(InitRange.getBegin(), "{"));
      FixIts.push_back(FixItHint::CreateInsertion(InitRange.getEnd(), "}"));
    }
  }

  if (!replaceDecl(ATL, Var, UseCTAD, Context, FixIts))
    return {};

  std::optional<FixItHint> IncludeFixIt =
      IncludeInserter.createIncludeInsertion(SM.getFileID(Var->getBeginLoc()),
                                             "<array>");
  if (IncludeFixIt)
    FixIts.push_back(std::move(*IncludeFixIt));

  return FixIts;
}

void AvoidCArraysCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ArrayType = Result.Nodes.getNodeAs<TypeLoc>("typeloc");

  bool IsVLA = ArrayType->getTypePtr()->isVariableArrayType();
  auto Diag = diag(ArrayType->getBeginLoc(),
                   "do not declare %select{C-style|C VLA}0 arrays, use "
                   "%select{std::array<>|std::vector<>}0 instead")
              << IsVLA;

  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var");
  const auto *VarDeclStmt = Result.Nodes.getNodeAs<DeclStmt>("decl");
  if (!Var || !VarDeclStmt)
    return;

  // FIXME: Support variables with attributes
  if (Var->hasAttrs())
    return;

  ArrayTypeLoc ATL = ArrayType->getUnqualifiedLoc().getAs<ArrayTypeLoc>();
  if (!ATL)
    return;
  Diag << replaceArray(ATL, VarDeclStmt, Var, *Result.Context);
}

void AvoidCArraysCheck::registerPPCallbacks(const SourceManager &SM,
                                            Preprocessor *PP,
                                            Preprocessor *ModuleExpanderPP) {
  this->PP = PP;
  IncludeInserter.registerPreprocessor(PP);
}

void AvoidCArraysCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

AvoidCArraysCheck::AvoidCArraysCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

} // namespace clang::tidy::modernize
