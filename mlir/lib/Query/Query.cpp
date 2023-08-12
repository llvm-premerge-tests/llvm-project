//===---- Query.cpp - mlir-query query ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Query.h"
#include "mlir/Query/Matcher/MatchFinder.h"
#include "mlir/Query/QuerySession.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::query {

static void printMatch(llvm::raw_ostream &os, QuerySession &qs, Operation *op,
                       const std::string &binding) {
  auto fileLoc = op->getLoc()->findInstanceOf<FileLineColLoc>();
  auto smloc = qs.sourceMgr->FindLocForLineAndColumn(
      qs.bufferId, fileLoc.getLine(), fileLoc.getColumn());
  qs.sourceMgr->PrintMessage(os, smloc, llvm::SourceMgr::DK_Note,
                             "\"" + binding + "\" binds here");
}

Query::~Query() = default;

mlir::LogicalResult InvalidQuery::run(llvm::raw_ostream &os,
                                      QuerySession &qs) const {
  os << errStr << "\n";
  return mlir::failure();
}

mlir::LogicalResult NoOpQuery::run(llvm::raw_ostream &os,
                                   QuerySession &qs) const {
  return mlir::success();
}

mlir::LogicalResult HelpQuery::run(llvm::raw_ostream &os,
                                   QuerySession &qs) const {
  os << "Available commands:\n\n"
        "  match MATCHER, m MATCHER      "
        "Match the mlir against the given matcher.\n\n";
  return mlir::success();
}

mlir::LogicalResult MatchQuery::run(llvm::raw_ostream &os,
                                    QuerySession &qs) const {
  int matchCount = 0;
  std::vector<Operation *> matches =
      matcher::MatchFinder().getMatches(qs.rootOp, matcher);
  os << "\n";
  for (Operation *op : matches) {
    os << "Match #" << ++matchCount << ":\n\n";
    // Placeholder "root" binding for the initial draft.
    printMatch(os, qs, op, "root");
  }
  os << matchCount << (matchCount == 1 ? " match.\n\n" : " matches.\n\n");

  return mlir::success();
}

} // namespace mlir::query
