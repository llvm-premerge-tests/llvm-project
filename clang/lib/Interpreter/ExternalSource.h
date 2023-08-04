//==----- ExternalSource.h - External AST Source for Code Completion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines components that make declarations parsed and executed by
// the interpreter visible to the context where code completion is being
// triggered.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExternalASTSource.h"

namespace clang {
class ASTContext;
class FileManager;
class ASTImporter;

class ExternalSource : public clang::ExternalASTSource {
  ASTContext &ChildASTCtxt;
  TranslationUnitDecl *ChildTUDeclCtxt;
  ASTContext &ParentASTCtxt;
  TranslationUnitDecl *ParentTUDeclCtxt;

  std::unique_ptr<ASTImporter> Importer;

public:
  ExternalSource(ASTContext &ChildASTCtxt, FileManager &ChildFM,
                 ASTContext &ParentASTCtxt, FileManager &ParentFM);
  bool FindExternalVisibleDeclsByName(const DeclContext *DC,
                                      DeclarationName Name) override;
  void
  completeVisibleDeclsMap(const clang::DeclContext *childDeclContext) override;
};
} // namespace clang
