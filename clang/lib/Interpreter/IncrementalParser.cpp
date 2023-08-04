//===--------- IncrementalParser.cpp - Incremental Compilation  -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class which performs incremental code compilation.
//
//===----------------------------------------------------------------------===//

#include "IncrementalParser.h"
#include "ExternalSource.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Timer.h"

#include <sstream>

namespace clang {

class IncrementalASTConsumer final : public ASTConsumer {
  Interpreter &Interp;
  std::unique_ptr<ASTConsumer> Consumer;

public:
  IncrementalASTConsumer(Interpreter &InterpRef, std::unique_ptr<ASTConsumer> C)
      : Interp(InterpRef), Consumer(std::move(C)) {}

  bool HandleTopLevelDecl(DeclGroupRef DGR) override final {
    if (DGR.isNull())
      return true;
    if (!Consumer)
      return true;

    for (Decl *D : DGR)
      if (auto *TSD = llvm::dyn_cast<TopLevelStmtDecl>(D);
          TSD && TSD->isSemiMissing())
        TSD->setStmt(Interp.SynthesizeExpr(cast<Expr>(TSD->getStmt())));

    return Consumer->HandleTopLevelDecl(DGR);
  }
  void HandleTranslationUnit(ASTContext &Ctx) override final {
    Consumer->HandleTranslationUnit(Ctx);
  }
  void HandleInlineFunctionDefinition(FunctionDecl *D) override final {
    Consumer->HandleInlineFunctionDefinition(D);
  }
  void HandleInterestingDecl(DeclGroupRef D) override final {
    Consumer->HandleInterestingDecl(D);
  }
  void HandleTagDeclDefinition(TagDecl *D) override final {
    Consumer->HandleTagDeclDefinition(D);
  }
  void HandleTagDeclRequiredDefinition(const TagDecl *D) override final {
    Consumer->HandleTagDeclRequiredDefinition(D);
  }
  void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) override final {
    Consumer->HandleCXXImplicitFunctionInstantiation(D);
  }
  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) override final {
    Consumer->HandleTopLevelDeclInObjCContainer(D);
  }
  void HandleImplicitImportDecl(ImportDecl *D) override final {
    Consumer->HandleImplicitImportDecl(D);
  }
  void CompleteTentativeDefinition(VarDecl *D) override final {
    Consumer->CompleteTentativeDefinition(D);
  }
  void CompleteExternalDeclaration(VarDecl *D) override final {
    Consumer->CompleteExternalDeclaration(D);
  }
  void AssignInheritanceModel(CXXRecordDecl *RD) override final {
    Consumer->AssignInheritanceModel(RD);
  }
  void HandleCXXStaticMemberVarInstantiation(VarDecl *D) override final {
    Consumer->HandleCXXStaticMemberVarInstantiation(D);
  }
  void HandleVTable(CXXRecordDecl *RD) override final {
    Consumer->HandleVTable(RD);
  }
  ASTMutationListener *GetASTMutationListener() override final {
    return Consumer->GetASTMutationListener();
  }
  ASTDeserializationListener *GetASTDeserializationListener() override final {
    return Consumer->GetASTDeserializationListener();
  }
  void PrintStats() override final { Consumer->PrintStats(); }
  bool shouldSkipFunctionBody(Decl *D) override final {
    return Consumer->shouldSkipFunctionBody(D);
  }
  static bool classof(const clang::ASTConsumer *) { return true; }
};

/// A custom action enabling the incremental processing functionality.
///
/// The usual \p FrontendAction expects one call to ExecuteAction and once it
/// sees a call to \p EndSourceFile it deletes some of the important objects
/// such as \p Preprocessor and \p Sema assuming no further input will come.
///
/// \p IncrementalAction ensures it keep its underlying action's objects alive
/// as long as the \p IncrementalParser needs them.
///
class IncrementalAction : public WrapperFrontendAction {
private:
  bool IsTerminating = false;
  std::vector<CodeCompletionResult>& CCResults;
  const CompilerInstance *ParentCI;

public:
  IncrementalAction(CompilerInstance &CI, llvm::LLVMContext &LLVMCtx,
                    llvm::Error &Err,
                    const CompilerInstance *ParentCI,
                    std::vector<CodeCompletionResult>& CCResults)
      : WrapperFrontendAction([&]() {
          llvm::ErrorAsOutParameter EAO(&Err);
          std::unique_ptr<FrontendAction> Act;
          switch (CI.getFrontendOpts().ProgramAction) {
          default:
            Err = llvm::createStringError(
                std::errc::state_not_recoverable,
                "Driver initialization failed. "
                "Incremental mode for action %d is not supported",
                CI.getFrontendOpts().ProgramAction);
            return Act;
          case frontend::ASTDump:
            [[fallthrough]];
          case frontend::ASTPrint:
            [[fallthrough]];
          case frontend::ParseSyntaxOnly:
            Act = CreateFrontendAction(CI);
            break;
          case frontend::PluginAction:
            [[fallthrough]];
          case frontend::EmitAssembly:
            [[fallthrough]];
          case frontend::EmitBC:
            [[fallthrough]];
          case frontend::EmitObj:
            [[fallthrough]];
          case frontend::PrintPreprocessedInput:
            [[fallthrough]];
          case frontend::EmitLLVMOnly:
            Act.reset(new EmitLLVMOnlyAction(&LLVMCtx));
            break;
          }
          return Act;
        }()),
        CCResults(CCResults), ParentCI(ParentCI){}
  FrontendAction *getWrapped() const { return WrappedAction.get(); }
  TranslationUnitKind getTranslationUnitKind() override {
    return TU_Incremental;
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    assert(CI.hasPreprocessor() && "No PP!");

    if (ParentCI) {
      // in code completion mode,
      CI.getPreprocessorOpts().SingleFileParseMode = true;

      CI.getLangOpts().SpellChecking = false;
      CI.getLangOpts().DelayedTemplateParsing = false;
      CI.setCodeCompletionConsumer(new ReplCompletionConsumer(CCResults));
    }
    // Use a code completion consumer?
    CodeCompleteConsumer *CompletionConsumer = nullptr;
    if (CI.hasCodeCompletionConsumer())
      CompletionConsumer = &CI.getCodeCompletionConsumer();

    Preprocessor &PP = CI.getPreprocessor();
    PP.EnterMainSourceFile();

    if (ParentCI) {
      ExternalSource *myExternalSource = new ExternalSource(
          CI.getASTContext(), CI.getFileManager(), ParentCI->getASTContext(),
          ParentCI->getFileManager());
      llvm::IntrusiveRefCntPtr<ExternalASTSource> astContextExternalSource(
          myExternalSource);
      CI.getASTContext().setExternalSource(astContextExternalSource);
      CI.getASTContext().getTranslationUnitDecl()->setHasExternalVisibleStorage(
          true);
    }

    if (!CI.hasSema())
      CI.createSema(getTranslationUnitKind(), CompletionConsumer);
  }

  // Do not terminate after processing the input. This allows us to keep various
  // clang objects alive and to incrementally grow the current TU.
  void EndSourceFile() override {
    // The WrappedAction can be nullptr if we issued an error in the ctor.
    if (IsTerminating && getWrapped())
      WrapperFrontendAction::EndSourceFile();
  }

  void FinalizeAction() {
    assert(!IsTerminating && "Already finalized!");
    IsTerminating = true;
    EndSourceFile();
  }
};

CodeGenerator *IncrementalParser::getCodeGen() const {
  FrontendAction *WrappedAct = Act->getWrapped();
  if (!WrappedAct->hasIRSupport())
    return nullptr;
  return static_cast<CodeGenAction *>(WrappedAct)->getCodeGenerator();
}

IncrementalParser::IncrementalParser() {}

IncrementalParser::IncrementalParser(Interpreter &Interp,
                                     std::unique_ptr<CompilerInstance> Instance,
                                     llvm::LLVMContext &LLVMCtx,
                                     llvm::Error &Err,
                                     const CompilerInstance *ParentCI,
                                     std::vector<CodeCompletionResult>& CCResults)
    : CI(std::move(Instance)) {
  llvm::ErrorAsOutParameter EAO(&Err);
  Act = std::make_unique<IncrementalAction>(*CI, LLVMCtx, Err, ParentCI, CCResults);
  if (Err)
    return;
  CI->ExecuteAction(*Act);
  std::unique_ptr<ASTConsumer> IncrConsumer =
      std::make_unique<IncrementalASTConsumer>(Interp, CI->takeASTConsumer());
  CI->setASTConsumer(std::move(IncrConsumer));
  Consumer = &CI->getASTConsumer();
  P.reset(
      new Parser(CI->getPreprocessor(), CI->getSema(), /*SkipBodies=*/false));
  P->Initialize();

  // An initial PTU is needed as CUDA includes some headers automatically
  auto PTU = ParseOrWrapTopLevelDecl();
  if (auto E = PTU.takeError()) {
    consumeError(std::move(E)); // FIXME
    return;                     // PTU.takeError();
  }

  if (CodeGenerator *CG = getCodeGen()) {
    std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
    CG->StartModule("incr_module_" + std::to_string(PTUs.size()),
                    M->getContext());
    PTU->TheModule = std::move(M);
    assert(PTU->TheModule && "Failed to create initial PTU");
  }
}

IncrementalParser::~IncrementalParser() {
  P.reset();
  Act->FinalizeAction();
}

llvm::Expected<PartialTranslationUnit &>
IncrementalParser::ParseOrWrapTopLevelDecl() {
  // Recover resources if we crash before exiting this method.
  Sema &S = CI->getSema();
  llvm::CrashRecoveryContextCleanupRegistrar<Sema> CleanupSema(&S);
  Sema::GlobalEagerInstantiationScope GlobalInstantiations(S, /*Enabled=*/true);
  Sema::LocalEagerInstantiationScope LocalInstantiations(S);

  PTUs.emplace_back(PartialTranslationUnit());
  PartialTranslationUnit &LastPTU = PTUs.back();
  // Add a new PTU.
  ASTContext &C = S.getASTContext();
  C.addTranslationUnitDecl();
  LastPTU.TUPart = C.getTranslationUnitDecl();

  // Skip previous eof due to last incremental input.
  if (P->getCurToken().is(tok::annot_repl_input_end)) {
    P->ConsumeAnyToken();
    // FIXME: Clang does not call ExitScope on finalizing the regular TU, we
    // might want to do that around HandleEndOfTranslationUnit.
    P->ExitScope();
    S.CurContext = nullptr;
    // Start a new PTU.
    P->EnterScope(Scope::DeclScope);
    S.ActOnTranslationUnitScope(P->getCurScope());
  }

  Parser::DeclGroupPtrTy ADecl;
  Sema::ModuleImportState ImportState;
  for (bool AtEOF = P->ParseFirstTopLevelDecl(ADecl, ImportState); !AtEOF;
       AtEOF = P->ParseTopLevelDecl(ADecl, ImportState)) {
    if (ADecl && !Consumer->HandleTopLevelDecl(ADecl.get()))
      return llvm::make_error<llvm::StringError>("Parsing failed. "
                                                 "The consumer rejected a decl",
                                                 std::error_code());
  }

  DiagnosticsEngine &Diags = getCI()->getDiagnostics();
  if (Diags.hasErrorOccurred()) {
    PartialTranslationUnit MostRecentPTU = {C.getTranslationUnitDecl(),
                                            nullptr};
    CleanUpPTU(MostRecentPTU);

    Diags.Reset(/*soft=*/true);
    Diags.getClient()->clear();
    return llvm::make_error<llvm::StringError>("Parsing failed.",
                                               std::error_code());
  }

  // Process any TopLevelDecls generated by #pragma weak.
  for (Decl *D : S.WeakTopLevelDecls()) {
    DeclGroupRef DGR(D);
    Consumer->HandleTopLevelDecl(DGR);
  }

  LocalInstantiations.perform();
  GlobalInstantiations.perform();

  Consumer->HandleTranslationUnit(C);

  return LastPTU;
}

std::pair<FileID, SourceLocation>
IncrementalParser::createSourceFile(llvm::StringRef SourceName,
                                    llvm::StringRef Input) {
  // Create an uninitialized memory buffer, copy code in and append "\n"
  size_t InputSize = Input.size(); // don't include trailing 0
  // MemBuffer size should *not* include terminating zero
  std::unique_ptr<llvm::MemoryBuffer> MB(
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(InputSize + 1,
                                                        SourceName.str()));
  char *MBStart = const_cast<char *>(MB->getBufferStart());
  memcpy(MBStart, Input.data(), InputSize);
  MBStart[InputSize] = '\n';

  SourceManager &SM = CI->getSourceManager();

  // FIXME: Create SourceLocation, which will allow clang to order the overload
  // candidates for example
  SourceLocation NewLoc = SM.getLocForStartOfFile(SM.getMainFileID());


  const clang::FileEntry *FE = SM.getFileManager().getVirtualFile(
      SourceName.str(), InputSize, 0 /* mod time*/);
  SM.overrideFileContents(FE, std::move(MB));

  // Create FileID for the current buffer.
  FileID FID = SM.createFileID(FE, NewLoc, SrcMgr::C_User);
  return {FID, NewLoc};
}

bool IncrementalParser::isCodeCompletionEnabled(){
  return !(CI->getFrontendOpts().CodeCompletionAt.FileName.empty());
}

llvm::Expected<PartialTranslationUnit &>
IncrementalParser::Parse(llvm::StringRef input) {
  Preprocessor &PP = CI->getPreprocessor();
  assert(PP.isIncrementalProcessingEnabled() && "Not in incremental mode!?");
  std::ostringstream SourceName;

  if (isCodeCompletionEnabled()) {
    SourceName << CI->getFrontendOpts().CodeCompletionAt.FileName;
  } else {
    SourceName << "input_line_" << InputCount++;
  }

  auto [FID, SrcLoc] = createSourceFile(SourceName.str(), input);

  if (isCodeCompletionEnabled()) {
    // createCodeCompletionConsumer enables the code completion point, which
    // must happen after the source file is created.
    CI->createCodeCompletionConsumer();
  }

  // NewLoc only used for diags.
  if (PP.EnterSourceFile(FID, /*DirLookup=*/nullptr, SrcLoc))
    return llvm::make_error<llvm::StringError>("Parsing failed. "
                                               "Cannot enter source file.",
                                               std::error_code());

  auto PTU = ParseOrWrapTopLevelDecl();

  if (!PTU) {
    return std::move(PTU.takeError());
  }

  if (isCodeCompletionEnabled()) {
    // there is no need to do extra lexing for code completion
    return PTU;
  }

  if (PP.getLangOpts().DelayedTemplateParsing) {
    // Microsoft-specific:
    // Late parsed templates can leave unswallowed "macro"-like tokens.
    // They will seriously confuse the Parser when entering the next
    // source file. So lex until we are EOF.
    Token Tok;
    do {
      PP.Lex(Tok);
    } while (Tok.isNot(tok::annot_repl_input_end));
  } else {
    Token AssertTok;
    PP.Lex(AssertTok);
    assert(AssertTok.is(tok::annot_repl_input_end) &&
           "Lexer must be EOF when starting incremental parse!");
  }

  if (std::unique_ptr<llvm::Module> M = GenModule())
    PTU->TheModule = std::move(M);

  return PTU;
}

std::unique_ptr<llvm::Module> IncrementalParser::GenModule() {
  static unsigned ID = 0;
  if (CodeGenerator *CG = getCodeGen()) {
    std::unique_ptr<llvm::Module> M(CG->ReleaseModule());
    CG->StartModule("incr_module_" + std::to_string(ID++), M->getContext());
    return M;
  }
  return nullptr;
}

void IncrementalParser::CleanUpPTU(PartialTranslationUnit &PTU) {
  TranslationUnitDecl *MostRecentTU = PTU.TUPart;
  TranslationUnitDecl *FirstTU = MostRecentTU->getFirstDecl();
  if (StoredDeclsMap *Map = FirstTU->getPrimaryContext()->getLookupPtr()) {
    for (auto I = Map->begin(); I != Map->end(); ++I) {
      StoredDeclsList &List = I->second;
      DeclContextLookupResult R = List.getLookupResult();
      for (NamedDecl *D : R) {
        if (D->getTranslationUnitDecl() == MostRecentTU) {
          List.remove(D);
        }
      }
      if (List.isNull())
        Map->erase(I);
    }
  }
}

llvm::StringRef IncrementalParser::GetMangledName(GlobalDecl GD) const {
  CodeGenerator *CG = getCodeGen();
  assert(CG);
  return CG->GetMangledName(GD);
}

} // end namespace clang
