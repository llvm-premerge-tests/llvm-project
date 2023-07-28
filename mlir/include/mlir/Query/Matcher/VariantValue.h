//===--- VariantValue.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Supports all the types required for dynamic Matcher construction.
// Used by the registry to construct matchers in a generic way.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H

#include "Diagnostics.h"
#include "MatchersInternal.h"

namespace mlir::query::matcher {

// Kind identifier that supports all types that VariantValue can contain.
class ArgKind {
public:
  enum Kind { AK_Matcher, AK_String };
  ArgKind(Kind k) : k(k) {}

  Kind getArgKind() const { return k; }

  bool operator<(const ArgKind &other) const { return k < other.k; }

  // String representation of the type.
  std::string asString() const;

private:
  Kind k;
};

// A variant matcher object to abstract simple and complex matchers into a
// single object type.
class VariantMatcher {
  class MatcherOps;

  // Payload interface to be specialized by each matcher type. It follows a
  // similar interface as VariantMatcher itself.
  class Payload {
  public:
    virtual ~Payload();
    virtual std::optional<DynMatcher> getDynMatcher() const = 0;
    virtual std::string getTypeAsString() const = 0;
  };

public:
  /// A null matcher.
  VariantMatcher();

  // Clones the provided matcher.
  static VariantMatcher SingleMatcher(DynMatcher matcher);

  // Makes the matcher the "null" matcher.
  void reset();

  // Checks if the matcher is null.
  bool isNull() const { return !value; }

  /// Returns the matcher
  std::optional<DynMatcher> getDynMatcher() const;

  // String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  explicit VariantMatcher(std::shared_ptr<Payload> value)
      : value(std::move(value)) {}

  class SinglePayload;

  std::shared_ptr<const Payload> value;
};

// Variant value class with a tagged union with value type semantics. It is used
// by the registry as the return value and argument type for the matcher factory
// methods. It can be constructed from any of the supported types:
//  - StringRef
//  - VariantMatcher
class VariantValue {
public:
  VariantValue() : type(VT_Nothing) {}

  VariantValue(const VariantValue &other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &other);

  // Specific constructors for each supported type.
  VariantValue(const StringRef string);
  VariantValue(const VariantMatcher &matcher);

  // String value functions.
  bool isString() const;
  const StringRef &getString() const;
  void setString(const StringRef &string);

  // Matcher value functions.
  bool isMatcher() const;
  const VariantMatcher &getMatcher() const;
  void setMatcher(const VariantMatcher &matcher);

  // String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  void reset();

  // All supported value types.
  enum ValueType {
    VT_Nothing,
    VT_String,
    VT_Matcher,
  };

  // All supported value types.
  union AllValues {
    StringRef *String;
    VariantMatcher *Matcher;
  };

  ValueType type;
  AllValues value;
};

// A VariantValue instance annotated with its parser context.
struct ParserValue {
  ParserValue() {}
  llvm::StringRef text;
  SourceRange range;
  VariantValue value;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_VARIANTVALUE_H
