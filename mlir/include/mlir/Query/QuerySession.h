//===--- QuerySession.h - mlir-query ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
#define MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H

#include "Query.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::query {

// Represents the state for a particular mlir-query session.
class QuerySession {
public:
  QuerySession(Operation *rootOp,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
               unsigned bufferId)
      : rootOp(rootOp), sourceMgr(sourceMgr), bufferId(bufferId),
        terminate(false) {}

  const std::shared_ptr<llvm::SourceMgr> &getSourceManager() {
    return sourceMgr;
  }

  Operation *rootOp;
  const std::shared_ptr<llvm::SourceMgr> sourceMgr;
  unsigned bufferId;
  bool terminate;
  llvm::StringMap<matcher::VariantValue> namedValues;
};

} // namespace mlir::query

#endif // MLIR_TOOLS_MLIRQUERY_QUERYSESSION_H
