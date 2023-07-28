//===---- Query.cpp - mlir-query query --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Query.h"
#include "mlir/Query/Matcher/MatchFinder.h"
#include "mlir/Query/QuerySession.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::query {

static void printMatch(llvm::raw_ostream &OS, QuerySession &QS, Operation *op,
                       std::string binding) {
  auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
  auto smloc = QS.sourceMgr->FindLocForLineAndColumn(
      QS.bufferId, fileLoc.getLine(), fileLoc.getColumn());
  QS.sourceMgr->PrintMessage(OS, smloc, llvm::SourceMgr::DK_Note,
                             "\"" + binding + "\" binds here");
}

Query::~Query() {}

bool InvalidQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << errStr << "\n";
  return false;
}

bool NoOpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  return true;
}

bool HelpQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  OS << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n\n";
  return true;
}

bool MatchQuery::run(llvm::raw_ostream &OS, QuerySession &QS) const {
  int matchCount = 0;
  std::vector<Operation *> matches =
      matcher::MatchFinder().getMatches(QS.rootOp, matcher);
  OS << "\n";
  for (Operation *op : matches) {
    OS << "Match #" << ++matchCount << ":\n\n";
    // Placeholder "root" binding for the initial draft.
    printMatch(OS, QS, op, "root");
  }
  OS << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");

  return true;
}

} // namespace mlir::query
