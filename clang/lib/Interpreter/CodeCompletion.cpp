//===------ CodeCompletion.cpp - Code Completion for ClangRepl -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the classes which performs code completion at the REPL.
//
//===----------------------------------------------------------------------===//

#include "clang/Interpreter/CodeCompletion.h"
#include "clang/AST/ASTImporter.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExternalASTSource.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "clang/Sema/CodeCompleteOptions.h"
#include "clang/Sema/Sema.h"

namespace clang {

const std::string CodeCompletionFileName = "input_line_[Completion]";

clang::CodeCompleteOptions getClangCompleteOpts() {
  clang::CodeCompleteOptions Opts;
  Opts.IncludeCodePatterns = true;
  Opts.IncludeMacros = true;
  Opts.IncludeGlobals = true;
  Opts.IncludeBriefComments = true;
  return Opts;
}

class CodeCompletionSubContext {
public:
  virtual ~CodeCompletionSubContext(){};
  virtual void
  HandleCodeCompleteResults(class Sema &S, CodeCompletionResult *InResults,
                            unsigned NumResults,
                            std::vector<std::string> &Results) = 0;
};

class ReplCompletionConsumer : public CodeCompleteConsumer {
public:
  ReplCompletionConsumer(std::vector<std::string> &Results,
                         std::unique_ptr<CodeCompletionSubContext> SubCtxt)
      : CodeCompleteConsumer(getClangCompleteOpts()),
        CCAllocator(std::make_shared<GlobalCodeCompletionAllocator>()),
        CCTUInfo(CCAllocator), Results(Results), SubCtxt(std::move(SubCtxt)) {
 }

  void ProcessCodeCompleteResults(class Sema &S, CodeCompletionContext Context,
                                  CodeCompletionResult *InResults,
                                  unsigned NumResults) final;

  CodeCompletionAllocator &getAllocator() override { return *CCAllocator; }

  CodeCompletionTUInfo &getCodeCompletionTUInfo() override { return CCTUInfo; }

private:
  std::shared_ptr<GlobalCodeCompletionAllocator> CCAllocator;
  CodeCompletionTUInfo CCTUInfo;
  std::vector<std::string> &Results;
  std::unique_ptr<CodeCompletionSubContext> SubCtxt;
};

void ReplCompletionConsumer::ProcessCodeCompleteResults(
    class Sema &S, CodeCompletionContext Context,
    CodeCompletionResult *InResults, unsigned NumResults) {
  SubCtxt->HandleCodeCompleteResults(S, InResults, NumResults, Results);
}

class IncrementalSyntaxOnlyAction : public SyntaxOnlyAction {
  const CompilerInstance *ParentCI;

public:
  IncrementalSyntaxOnlyAction(const CompilerInstance *ParentCI)
      : ParentCI(ParentCI) {}

protected:
  void ExecuteAction() override;
};

class ExternalSource : public clang::ExternalASTSource {
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

// This method is intended to set up `ExternalASTSource` to the running
// compiler instance before the super `ExecuteAction` triggers parsing
void IncrementalSyntaxOnlyAction::ExecuteAction() {
  CompilerInstance &CI = getCompilerInstance();
  ExternalSource *myExternalSource =
      new ExternalSource(CI.getASTContext(), CI.getFileManager(),
                         ParentCI->getASTContext(), ParentCI->getFileManager());
  llvm::IntrusiveRefCntPtr<clang::ExternalASTSource> astContextExternalSource(
      myExternalSource);
  CI.getASTContext().setExternalSource(astContextExternalSource);
  CI.getASTContext().getTranslationUnitDecl()->setHasExternalVisibleStorage(
      true);

  SyntaxOnlyAction::ExecuteAction();
}

ExternalSource::ExternalSource(ASTContext &ChildASTCtxt, FileManager &ChildFM,
                               ASTContext &ParentASTCtxt, FileManager &ParentFM)
    : ChildTUDeclCtxt(ChildASTCtxt.getTranslationUnitDecl()),
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


class CCSubContextRegular : public CodeCompletionSubContext {
  StringRef prefix;

public:
  CCSubContextRegular(StringRef prefix) : prefix(prefix){};
  virtual ~CCSubContextRegular(){};
  void HandleCodeCompleteResults(
      class Sema &S, CodeCompletionResult *InResults, unsigned NumResults,
      std::vector<std::string> &Results) override;
};

class CCSubContextCallSite : public CodeCompletionSubContext {
  StringRef calleeName;
  StringRef Prefix;
  std::optional<const FunctionDecl *> lookUp(CodeCompletionResult *InResults,
                                             unsigned NumResults);

public:
  CCSubContextCallSite(StringRef calleeName, StringRef Prefix)
      : calleeName(calleeName), Prefix(Prefix) {}
  virtual ~CCSubContextCallSite(){};
  void HandleCodeCompleteResults(
      class Sema &S, CodeCompletionResult *InResults, unsigned NumResults,
      std::vector<std::string> &Results) override;
};

void CCSubContextRegular::HandleCodeCompleteResults(
    class Sema &S, CodeCompletionResult *InResults, unsigned NumResults,
    std::vector<std::string> &Results) {
  for (unsigned I = 0; I < NumResults; ++I) {
    auto &Result = InResults[I];
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (auto *ID = Result.Declaration->getIdentifier()) {
        Results.push_back(ID->getName().str());
      }
      break;
    case CodeCompletionResult::RK_Keyword:
      Results.push_back(Result.Keyword);
      break;
    default:
      break;
    }
  }
}

std::optional<const FunctionDecl *>
CCSubContextCallSite::lookUp(CodeCompletionResult *InResults,
                             unsigned NumResults) {
  for (unsigned I = 0; I < NumResults; I++) {
    auto &Result = InResults[I];
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (Result.Hidden) {
        continue;
      }
      if (const auto *Function = Result.Declaration->getAsFunction()) {
        if (Function->isDestroyingOperatorDelete() ||
            Function->isOverloadedOperator()) {
          continue;
        }

        auto Name = Function->getDeclName();
        switch (Name.getNameKind()) {
        case DeclarationName::CXXConstructorName:
        case DeclarationName::CXXDestructorName:
          continue;
        default:
          if (Function->getName() == calleeName) {
            return Function;
          }
          continue;
        }
      }
      break;
    default:
      break;
    }
  }
  return std::nullopt;
}

void CCSubContextCallSite::HandleCodeCompleteResults(
    class Sema &S, CodeCompletionResult *InResults, unsigned NumResults,
    std::vector<std::string> &Results) {
  auto Function = lookUp(InResults, NumResults);
  if (!Function)
    return;
  for (unsigned I = 0; I < NumResults; I++) {
    auto &Result = InResults[I];
    switch (Result.Kind) {
    case CodeCompletionResult::RK_Declaration:
      if (Result.Hidden) {
        continue;
      }
      if (!Result.Declaration->getIdentifier()) {
        continue;
      }
      if (auto *DD = dyn_cast<ValueDecl>(Result.Declaration)) {
        if (!DD->getName().startswith(Prefix))
          continue;

        auto ArgumentType = DD->getType();
        auto RequiredType = (*Function)->getParamDecl(0)->getType();
        if (RequiredType->isReferenceType()) {
          QualType RT = RequiredType->castAs<ReferenceType>()->getPointeeType();
          Sema::ReferenceConversions RefConv;
          Sema::ReferenceCompareResult RefRelationship =
            S.CompareReferenceRelationship(SourceLocation(), RT, ArgumentType,
                                             &RefConv);
          if (RefRelationship == Sema::Ref_Compatible) {
            Results.push_back(DD->getName().str());
          } else if (RefRelationship == Sema::Ref_Related) {
            Results.push_back(DD->getName().str());
          }
        } else if (S.Context.hasSameType(ArgumentType, RequiredType)) {
          Results.push_back(DD->getName().str());
        }
      }
      break;
    default:
      break;
    }
  }
}


static std::pair<StringRef, std::unique_ptr<CodeCompletionSubContext>>
getCodeCompletionSubContext(llvm::StringRef CurInput, size_t Pos) {
  size_t LeftParenPos = CurInput.rfind("(");
  if (LeftParenPos == llvm::StringRef::npos) {
    size_t space_pos = CurInput.rfind(" ");
    llvm::StringRef Prefix;
    if (space_pos == llvm::StringRef::npos) {
      Prefix = CurInput;
    } else {
      Prefix = CurInput.substr(space_pos + 1);
    }
    return {Prefix, std::move(std::make_unique<CCSubContextRegular>(Prefix))};
  }
  auto subs = CurInput.substr(0, LeftParenPos);
  size_t start_pos = subs.rfind(" ");
  if (start_pos == llvm::StringRef::npos) {
    start_pos = 0;
  }
  auto Prefix = CurInput.substr(LeftParenPos + 1, Pos);
  return {Prefix,
          std::make_unique<CCSubContextCallSite>(
              subs.substr(start_pos, LeftParenPos - start_pos), Prefix)};
}


void codeComplete(CompilerInstance *InterpCI, llvm::StringRef Content,
                  unsigned Line, unsigned Col, const CompilerInstance *ParentCI,
                  std::vector<std::string> &CCResults) {
  auto DiagOpts = DiagnosticOptions();
  auto Pos = Col - 1;
  auto [s, SubCtxt] = getCodeCompletionSubContext(Content, Pos);
  auto consumer = ReplCompletionConsumer(CCResults, std::move(SubCtxt));

  auto diag = InterpCI->getDiagnosticsPtr();
  std::unique_ptr<ASTUnit> AU(ASTUnit::LoadFromCompilerInvocationAction(
      InterpCI->getInvocationPtr(), std::make_shared<PCHContainerOperations>(),
      diag));
  llvm::SmallVector<clang::StoredDiagnostic, 8> sd = {};
  llvm::SmallVector<const llvm::MemoryBuffer *, 1> tb = {};
  InterpCI->getFrontendOpts().Inputs[0] = FrontendInputFile(
      CodeCompletionFileName, Language::CXX, InputKind::Source);
  auto Act = std::unique_ptr<IncrementalSyntaxOnlyAction>(
      new IncrementalSyntaxOnlyAction(ParentCI));
  std::unique_ptr<llvm::MemoryBuffer> MB =
      llvm::MemoryBuffer::getMemBufferCopy(Content, CodeCompletionFileName);
  llvm::SmallVector<ASTUnit::RemappedFile, 4> RemappedFiles;

  RemappedFiles.push_back(std::make_pair(CodeCompletionFileName, MB.get()));
  // we don't want the AU destructor to release the memory buffer that MB
  // owns twice, because MB handles its resource on its own.
  AU->setOwnsRemappedFileBuffers(false);
  AU->CodeComplete(CodeCompletionFileName, 1, Col, RemappedFiles, false, false,
                   false, consumer,
                   std::make_shared<clang::PCHContainerOperations>(), *diag,
                   InterpCI->getLangOpts(), InterpCI->getSourceManager(),
                   InterpCI->getFileManager(), sd, tb, std::move(Act));
}

} // namespace clang
