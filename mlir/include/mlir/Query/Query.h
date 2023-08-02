//===--- Query.h - mlir-query -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERY_H
#define MLIR_TOOLS_MLIRQUERY_QUERY_H

#include "Matcher/VariantValue.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace mlir::query {

enum class QueryKind { Invalid, NoOp, Help, Match };

class QuerySession;

struct Query : llvm::RefCountedBase<Query> {
  Query(QueryKind kind) : kind(kind) {}
  virtual ~Query();

  // Perform the query on QS and print output to OS.
  // Return false if an error occurs, otherwise return true.
  virtual bool run(llvm::raw_ostream &OS, QuerySession &QS) const = 0;

  llvm::StringRef remainingContent;
  const QueryKind kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

// Any query which resulted in a parse error. The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const llvm::Twine &errStr)
      : Query(QueryKind::Invalid), errStr(errStr.str()) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  std::string errStr;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Invalid;
  }
};

// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QueryKind::NoOp) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::NoOp;
  }
};

// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QueryKind::Help) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Help;
  }
};

// Query for "match MATCHER".
struct MatchQuery : Query {
  MatchQuery(StringRef source, const matcher::DynMatcher &matcher)
      : Query(QueryKind::Match), matcher(matcher), source(source) {}
  bool run(llvm::raw_ostream &OS, QuerySession &QS) const override;

  const matcher::DynMatcher matcher;

  StringRef source;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Match;
  }
};

} // namespace mlir::query

#endif
