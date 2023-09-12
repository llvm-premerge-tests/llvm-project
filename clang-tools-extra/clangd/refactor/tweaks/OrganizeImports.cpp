//===--- OrganizeImports.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncludeCleaner.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "clang-include-cleaner/IncludeSpeller.h"
#include "clang-include-cleaner/Types.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <string>
#include <vector>

namespace clang::clangd {
namespace {

// Tweak for applying IWYU-related changes (removing unused and adding missing
// includes) in batch. The tweak can be triggered via the "Organize Imports"
// source action (VS Code). The set of changes applied fully corresponds to the
// findings from the clang-include-cleaner tool.
class OrganizeImports : public Tweak {
public:
  const char *id() const override;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override {
    return "Remove unused and add missing includes";
  }
  llvm::StringLiteral kind() const override { return CodeAction::SOURCE_KIND; }
};
REGISTER_TWEAK(OrganizeImports)

bool OrganizeImports::prepare(const Tweak::Selection &Inputs) {
  if (Inputs.AST->getLangOpts().ObjC)
    return false;
  if (std::find(Inputs.RequestedActionKinds.begin(),
                Inputs.RequestedActionKinds.end(),
                CodeAction::SOURCE_KIND) != Inputs.RequestedActionKinds.end())
    return true;
  if (Inputs.RequestedActionKinds.empty())
    // To accommodate clients without knowledge of source actions, we trigger
    // without checking code action kinds when inside the preamble region.
    return offsetToPosition(Inputs.Code, Inputs.SelectionEnd) <=
           offsetToPosition(Inputs.Code, Inputs.AST->getPreambleBounds().Size);
  return false;
}

Expected<Tweak::Effect> OrganizeImports::apply(const Selection &Inputs) {
  IncludeCleanerFindings Findings = computeIncludeCleanerFindings(*Inputs.AST);
  const auto MainFilePath = Inputs.AST->tuPath();
  tooling::Replacements Replacements;
  for (const auto *Inc : Findings.UnusedIncludes)
    llvm::cantFail(Replacements.add(
        tooling::Replacement{MainFilePath, UINT_MAX, 1, Inc->Written}));

  const auto &SM = Inputs.AST->getSourceManager();
  llvm::DenseSet<include_cleaner::Header> Providers;
  for (const auto &Missing : Findings.MissingIncludes) {
    assert(!Missing.Providers.empty());
    Providers.insert(Missing.Providers[0]);
  }

  for (const auto &P : Providers) {
    std::string Spelling = include_cleaner::spellHeader(
        {P, Inputs.AST->getPreprocessor().getHeaderSearchInfo(),
         SM.getFileEntryForID(SM.getMainFileID())});
    llvm::cantFail(Replacements.add(tooling::Replacement{
        MainFilePath, UINT_MAX, 0, "#include " + Spelling}));
  }

  auto FileStyle =
      format::getStyle(format::DefaultFormatStyle, MainFilePath,
                       format::DefaultFallbackStyle, Inputs.Code, Inputs.FS);
  if (!FileStyle) {
    elog("Couldn't get style for {0}: {1}", MainFilePath,
         FileStyle.takeError());
    FileStyle = format::getLLVMStyle();
  }
  auto Final =
      format::cleanupAroundReplacements(Inputs.Code, Replacements, *FileStyle);
  if (!Final)
    return Final.takeError();
  if (Final->empty())
    return Tweak::Effect{"No edits to apply.", {}};
  return Effect::mainFileEdit(SM, *Final);
}

} // namespace
} // namespace clang::clangd
