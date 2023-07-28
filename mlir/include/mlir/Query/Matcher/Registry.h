//===--- Registry.h - Matcher registry --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry of all known matchers.
//
// The registry provides a generic interface to construct any matcher by name.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H

#include "Diagnostics.h"
#include "Marshallers.h"
#include "VariantValue.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::query::matcher {

using MatcherCtor = const internal::MatcherDescriptor *;

struct MatcherCompletion {
  MatcherCompletion() = default;
  MatcherCompletion(llvm::StringRef typedText, llvm::StringRef matcherDecl)
      : typedText(typedText.str()), matcherDecl(matcherDecl.str()) {}

  bool operator==(const MatcherCompletion &other) const {
    return typedText == other.typedText && matcherDecl == other.matcherDecl;
  }

  // The text to type to select this matcher.
  std::string typedText;

  // The "declaration" of the matcher, with type information.
  std::string matcherDecl;
};

class Registry {
public:
  Registry() = delete;

  static std::optional<MatcherCtor>
  lookupMatcherCtor(llvm::StringRef matcherName);

  static std::vector<ArgKind> getAcceptedCompletionTypes(
      llvm::ArrayRef<std::pair<MatcherCtor, unsigned>> context);

  static std::vector<MatcherCompletion>
  getMatcherCompletions(ArrayRef<ArgKind> acceptedTypes);

  static VariantMatcher constructMatcher(MatcherCtor ctor,
                                         SourceRange nameRange,
                                         ArrayRef<ParserValue> args,
                                         Diagnostics *error);
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_REGISTRY_H
