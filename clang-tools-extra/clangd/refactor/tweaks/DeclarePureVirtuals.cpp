//===--- DeclarePureVirtuals.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Selection.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "clang/AST/ASTContextAllocate.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/AttributeCommonInfo.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/ASTSelection.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>
#include <unordered_set>

namespace clang {
namespace clangd {
namespace {

// TODO: access control

// TODO: doc
// void reprint(const CXXMethodDecl &M, llvm::raw_string_ostream &OS,
//              const PrintingPolicy &PP) {
//   std::string Declarator;
//   {
//     llvm::raw_string_ostream OS(Declarator);
//     const char *Sep = "";

//     // PrintingPolicy UnparenPP(PP);
//     // UnparenPP.ParenthesizeFunctionName = false;
//     // OS << printType(M.getType(), *M.getDeclContext(), M.getName(),
//     // &UnparenPP) << "(";

//     OS << M.getDeclName() << "(";
//     for (const auto &Param : M.parameters()) {
//       OS << Sep;
//       Param->print(OS, PP);
//       Sep = ", ";
//     }
//     OS << ")";
//   }
//   M.getReturnType().print(OS, PP, Declarator);
//   M.getMethodQualifiers().print(OS, PP, /*appendSpaceIfNonEmpty=*/true);
//   switch (M.getRefQualifier()) {
//   case RQ_None:
//     break;
//   case RQ_LValue:
//     OS << " &";
//     break;
//   case RQ_RValue:
//     OS << " &&";
//     break;
//   }
//   OS << " override;\n";
// }

// void reprintWrapper(std::string &To, const CXXMethodDecl &Method,
//                     PrintingPolicy pp, const CXXRecordDecl *Class) {
//   struct HandleScope : public PrintingCallbacks {
//     const CXXRecordDecl *Class;
//     mutable CXXBasePaths Paths;
//     HandleScope(const CXXRecordDecl *Class) : Class(Class), Paths() {}
//     virtual ~HandleScope() = default;

//     bool isScopeVisible(const clang::DeclContext *DC) const override {
//       if (DC->Encloses(Class))
//         return true;
//       if (const auto *MaybeBase = llvm::dyn_cast<CXXRecordDecl>(DC))
//         if (Class->isDerivedFrom(MaybeBase, Paths))
//           return true;
//       return false;
//     }
//   } PCallbacks(Class);

//   PrintingPolicy PP(pp);
//   PP.PolishForDeclaration = true;
//   PP.TerseOutput = true;
//   PP.Callbacks = &PCallbacks;
//   std::string Code;
//   llvm::raw_string_ostream OS(Code);
//   reprint(Method, OS, PP);
//   To.append(OS.str());
// }

// Determines if the class Cls which provided the final overrider map contains
// itself (not inherited) an overrider of Method.
bool hasOverrideFor(const CXXRecordDecl &Cls, const CXXFinalOverriderMap &Map,
                    const CXXMethodDecl *Method) {
  const auto *const It = Map.find(Method);
  if (It == Map.end()) {
    return false;
  }

  const auto &SubobjectsWithOverrider = It->second;
  for (const auto &[SubobjectId, FinalOverridersPerSubobj] :
       SubobjectsWithOverrider) {
    for (const UniqueVirtualMethod &FinalOverriderForSubobj :
         FinalOverridersPerSubobj) {
      if (FinalOverriderForSubobj.Method->getDeclContext() == &Cls)
        return true;
    }
  }

  return false;
}

// Appends a declaration to To which overrides Method.
void appendDeclForOverride(std::string &To, const CXXMethodDecl *Method,
                           const SourceManager &SM,
                           const syntax::TokenBuffer &TokBuf) {
  const SourceRange MethodDeclRange{Method->getBeginLoc(), Method->getEndLoc()};

  const llvm::ArrayRef<syntax::Token> Tokens =
      TokBuf.expandedTokens(MethodDeclRange);
  const auto EqTok =
      llvm::find_if(llvm::reverse(Tokens), [](const syntax::Token &Tok) {
        return Tok.kind() == tok::equal;
      });
  assert(EqTok != Tokens.rend());

  const syntax::Token *VirtTok =
      llvm::find_if(Tokens, [](const syntax::Token &Tok) {
        return Tok.kind() == tok::kw_virtual;
      });
  const syntax::Token *PostVirtTok = VirtTok;
  if (VirtTok == Tokens.end()) {
    PostVirtTok = Tokens.begin();
  } else {
    ++PostVirtTok;
  }

  assert(!Tokens.empty());
  // If we copy each token individually, we skip over whitespaces and comments.
  // Therefore, we copy entire source ranges between the tokens to remove.
  // TODO: we can't easily skip attributes since they consist of multiple
  // tokens, including balanced parentheses
  To.append(SM.getCharacterData(Tokens.begin()->location()),
            SM.getCharacterData(VirtTok->location()));
  To.append(SM.getCharacterData(PostVirtTok->location()),
            SM.getCharacterData(EqTok->location()));

  if (To.back() != ' ')
    To.push_back(' ');
  To.append("override;\n");
}

// TODO: doc
struct Additions {
  std::string AllAdditions;
  size_t Cutoff = 0;
  int NumAdditions = 0;
};

// TODO: remove
void dumpFinalOverriderMap(const CXXFinalOverriderMap &Fom,
                           llvm::raw_ostream &os) {
  for (const auto &[Method, SubobjectsWithOverrider] : Fom) {
    os << "Method " << Method->getQualifiedNameAsString() << ":\n";
    for (const auto &[SubobjectId, FinalOverridersPerSubobj] :
         SubobjectsWithOverrider) {
      os << "  Subobject " << SubobjectId << ":\n";
      for (const UniqueVirtualMethod &FinalOverriderForSubobj :
           FinalOverridersPerSubobj) {
        os << "    Final Overrider "
           << FinalOverriderForSubobj.Method->getQualifiedNameAsString()
           << "\n";
      }
    }
  }
}

// Returns a string with override declarations for all virtual functions in the
// inheritance hierarchy of Start (including Start itself) which are still pure
// virtual in Target. Start can be the same class as Target. Target must be the
// same as or derived from Start.
Additions collectPureVirtualFuncOverrideDecls(const CXXRecordDecl &Target,
                                              const CXXRecordDecl &Start,
                                              const SourceManager &SM,
                                              const syntax::TokenBuffer &TokBuf,
                                              PrintingPolicy PP,
                                              int MaxAdditionsToPreview) {
  Additions Additions;
  // If the inheritance forms a diamond, we can inherit the same virtual
  // function from multiple intermediate base classes. Those can themselves
  // re-declare the function as pure:
  //   struct A     { virtual void foo() = 0; };
  //   struct B : A { virtual void foo() = 0; };
  //   struct C : A { virtual void foo() = 0; };
  //   struct D : B, C {};
  // Since the final overrider map maps subobject -> final overriders, we will
  // therefore get:
  // - A::foo -> [(1, [B::foo]), (2, C::foo)]
  // - B::foo -> [(1, [B::foo])]
  // - C::foo -> [(2, [C::foo])]
  // Since both B::foo and C::foo are pure virtual, we need to make sure we only
  // emit one override for A::foo.
  std::unordered_set<const CXXMethodDecl *> SeenFinalOverriders;

  assert(&Target == &Start || Target.isDerivedFrom(&Start));

  CXXFinalOverriderMap FinalOverriderMapOfStart;
  Start.getFinalOverriders(FinalOverriderMapOfStart);

  CXXFinalOverriderMap FinalOverriderMapOfTarget;
  // TODO: is this true if the class re-declares a pure virtual function?
  // If &Target == &Start then Target doesn't already have any
  // overrides for functions that are pure in Start. The map remains empty,
  // which means hasOverrideFor will return false below, which is OK since the
  // function is then counted as pure virtual.
  if (&Target != &Start) {
    Target.getFinalOverriders(FinalOverriderMapOfTarget);
  }

  for (const auto &[Method, SubobjectsWithOverrider] :
       FinalOverriderMapOfStart) {
    bool AppendedOverriderForThisMethod = false;
    for (const auto &[SubobjectId, FinalOverridersPerSubobj] :
         SubobjectsWithOverrider) {
      // TODO: if FinalOverriders.length() > 1, abort?
      for (const UniqueVirtualMethod &FinalOverriderForSubobj :
           FinalOverridersPerSubobj) {
        if (FinalOverriderForSubobj.Method->isPure() &&
            !hasOverrideFor(Target, FinalOverriderMapOfTarget,
                            FinalOverriderForSubobj.Method) &&
            !AppendedOverriderForThisMethod &&
            !SeenFinalOverriders.count(FinalOverriderForSubobj.Method)) {
          // reprintWrapper(Additions.AllAdditions, *Override.Method, PP,
          // &Target);
          appendDeclForOverride(Additions.AllAdditions,
                                FinalOverriderForSubobj.Method, SM, TokBuf);
          if (Additions.NumAdditions++ == MaxAdditionsToPreview) {
            Additions.Cutoff = Additions.AllAdditions.size();
          }

          // If we inherit the same pure virtual function A::foo from multiple
          // base classes B, C, it could be pure virtual in B but implemented in
          // C. In that case, A::foo is still pure virtual in D:
          //   struct A     { virtual void foo() = 0;      };
          //   struct B : A { virtual void foo() = 0;      };
          //   struct C : A {         void foo() override; };
          //   struct D : B, C {}; // A::foo is still pure virtual in D for the
          //                       // B subobject
          // Therefore we still have to override A::foo in D to make it
          // non-abstract. Maybe we should emit some warning in that case, since
          // it's not obvious that we're overriding A::foo from C as well.
          AppendedOverriderForThisMethod = true;
        }
        // The FinalOverriderMap lists methods declarations from all base
        // classes, including indirect bases. Therefore we can encounter the
        // same final overrider multiple times, e.g. via an indirect base and
        // its derived class.
        SeenFinalOverriders.emplace(FinalOverriderForSubobj.Method);
      }
    }
  }

  return Additions;
}

// Finds the CXXBaseSpecifier in/under the selection, if any.
const CXXBaseSpecifier *findBaseSpecifier(const SelectionTree::Node *Node) {
  if (!Node)
    return nullptr;

  const DynTypedNode &ASTNode = Node->ASTNode;
  const CXXBaseSpecifier *BaseSpec = ASTNode.get<CXXBaseSpecifier>();
  if (BaseSpec)
    return BaseSpec;

  const SelectionTree::Node *const Parent = Node->Parent;
  if (Parent) {
    if (const CXXBaseSpecifier *BaseSpec =
            Parent->ASTNode.get<CXXBaseSpecifier>()) {
      return BaseSpec;
    }

    // This happens if Node is a RecordTypeLoc, e.g. when selecting the base
    // name in the base specifier
    if (auto const *Parent2 = Parent->Parent) {
      // not sure if this can ever be null
      return Parent2->ASTNode.get<CXXBaseSpecifier>();
    }
  }

  return nullptr;
}

// TODO: doc
llvm::Expected<Additions> generateOverrideDeclarations(
    const ParsedAST &AST, const CXXBaseSpecifier *SelectedBaseSpecifier,
    const CXXRecordDecl *SelectedDerivedClass, int MaxAdditionsToPreview) {
  const SourceManager &SM = AST.getSourceManager();

  const clang::PrintingPolicy PP = AST.getASTContext().getPrintingPolicy();
  if (SelectedBaseSpecifier) {
    // TODO: can getType return null?
    auto const *Start = SelectedBaseSpecifier->getType()->getAsCXXRecordDecl();
    if (!Start) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "selected base class specifier does not refer to a C++ class");
    }
    return collectPureVirtualFuncOverrideDecls(*SelectedDerivedClass, *Start,
                                               SM, AST.getTokens(), PP,
                                               MaxAdditionsToPreview);
  }

  return collectPureVirtualFuncOverrideDecls(
      *SelectedDerivedClass, *SelectedDerivedClass, SM, AST.getTokens(), PP,
      MaxAdditionsToPreview);
}

/// Declares overrides for all pure virtual function in a class hierarchy,
/// starting with a base class specifier.
///
/// Before:
///   class Base    { virtual void foo() = 0; };
///   class Derived {};
///         ^^^^^^^
///
/// After:
///   class Base    { virtual void foo() = 0; };
///   class Derived {         void foo() override; };
class DeclarePureVirtuals : public Tweak {
public:
  const char *id() const override;

  std::string title() const override {
    std::string_view Preview =
        std::string_view{Additions.AllAdditions}.substr(0, Additions.Cutoff);
    return llvm::formatv(
        "Override pure virtual function{0}:\n{1}{2}",
        Additions.NumAdditions == 1 ? "" : "s", Additions.AllAdditions, Preview,
        Additions.NumAdditions > MaxAdditionsToPreview ? "\n..." : "");
  }

  llvm::StringLiteral kind() const override {
    return CodeAction::GENERATE_KIND;
  }

  bool prepare(const Selection &Sel) override {
    if (!select(Sel))
      return false;

    auto Result = generateOverrideDeclarations(*Sel.AST, SelectedBaseSpecifier,
                                               SelectedDerivedClass,
                                               MaxAdditionsToPreview);
    if (Result.takeError()) {
      // TODO: report error? how?
      return false;
    }
    Additions = std::move(*Result);

    return Additions.NumAdditions > 0;
  }

  Expected<Effect> apply(const Selection &Sel) override {
    assert(SelectedDerivedClass); // prepare must have been called and returned
                                  // true

    SourceManager &SM = Sel.AST->getSourceManager();

    // TODO: can we apply this tweak in a header if we apply the effect to the
    // "main file"?
    return Effect::mainFileEdit(SM, tooling::Replacements(tooling::Replacement(
                                        SM, SelectedDerivedClass->getEndLoc(),
                                        0, Additions.AllAdditions)));
  }

private:
  // TODO: doc
  bool select(const Selection &Sel) {
    const SelectionTree::Node *const CommonAncestor =
        Sel.ASTSelection.commonAncestor();
    if (!CommonAncestor)
      return false;

    // maybe selected a class, in which case override functions of all bases
    SelectedDerivedClass = CommonAncestor->ASTNode.get<CXXRecordDecl>();
    if (SelectedDerivedClass) {
      // if this is a forward-declaration, accessing bases() will crash
      if (!SelectedDerivedClass->isCompleteDefinition())
        return false;
      return !SelectedDerivedClass->bases().empty();
    }

    // maybe selected a base class specifier, in which case only override those
    // bases's functions
    const DeclContext &DC = CommonAncestor->getDeclContext();
    SelectedDerivedClass = dyn_cast<CXXRecordDecl>(&DC);
    if (SelectedDerivedClass) {
      SelectedBaseSpecifier = findBaseSpecifier(CommonAncestor);
      return SelectedBaseSpecifier != nullptr;
    }

    return false;
  }

  const CXXRecordDecl *SelectedDerivedClass = nullptr;
  const CXXBaseSpecifier *SelectedBaseSpecifier = nullptr;
  Additions Additions;
  static constexpr inline int MaxAdditionsToPreview = 5;
};

REGISTER_TWEAK(DeclarePureVirtuals)

} // namespace
} // namespace clangd
} // namespace clang
