//===--- ExternalSource.cpp - External AST Source for Code Completion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file implements classes that make declarations parsed and executed by the
// interpreter visible to the context where code completion is being triggered.
//
//===----------------------------------------------------------------------===//

#include "ExternalSource.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/DeclarationName.h"
#include "clang/Basic/IdentifierTable.h"

namespace clang {
ExternalSource::ExternalSource(ASTContext &ChildASTCtxt, FileManager &ChildFM,
                               ASTContext &ParentASTCtxt, FileManager &ParentFM)
    : ChildASTCtxt(ChildASTCtxt),
      ChildTUDeclCtxt(ChildASTCtxt.getTranslationUnitDecl()),
      ParentASTCtxt(ParentASTCtxt),
      ParentTUDeclCtxt(ParentASTCtxt.getTranslationUnitDecl()) {
  ASTImporter *importer =
      new ASTImporter(ChildASTCtxt, ChildFM, ParentASTCtxt, ParentFM,
                      /*MinimalImport : ON*/ true);
  Importer.reset(importer);
}

bool ExternalSource::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                                    DeclarationName Name) {
  IdentifierTable &ParentIdTable = ParentASTCtxt.Idents;

  auto ParentDeclName =
      DeclarationName(&(ParentIdTable.get(Name.getAsString())));

  DeclContext::lookup_result lookup_result =
      ParentTUDeclCtxt->lookup(ParentDeclName);

  if (!lookup_result.empty()) {
    return true;
  }
  return false;
}

void ExternalSource::completeVisibleDeclsMap(
    const DeclContext *ChildDeclContext) {
  assert(ChildDeclContext && ChildDeclContext == ChildTUDeclCtxt &&
         "No child decl context!");

  if (!ChildDeclContext->hasExternalVisibleStorage())
    return;

  for (auto *DeclCtxt = ParentTUDeclCtxt; DeclCtxt != nullptr;
       DeclCtxt = DeclCtxt->getPreviousDecl()) {
    for (auto &IDeclContext : DeclCtxt->decls()) {
      if (NamedDecl *Decl = llvm::dyn_cast<NamedDecl>(IDeclContext)) {
        if (auto DeclOrErr = Importer->Import(Decl)) {
          if (NamedDecl *importedNamedDecl =
                  llvm::dyn_cast<NamedDecl>(*DeclOrErr)) {
            SetExternalVisibleDeclsForName(ChildDeclContext,
                                           importedNamedDecl->getDeclName(),
                                           importedNamedDecl);
          }

        } else {
          llvm::consumeError(DeclOrErr.takeError());
        }
      }
    }
    ChildDeclContext->setHasExternalLexicalStorage(false);
  }
}

} // namespace clang
