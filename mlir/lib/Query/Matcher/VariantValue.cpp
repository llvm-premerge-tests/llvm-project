//===--- MatcherVariantvalue.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/VariantValue.h"

namespace mlir::query::matcher {

std::string ArgKind::asString() const {
  switch (getArgKind()) {
  case AK_String:
    return "String";
  case AK_Matcher:
    return "Matcher";
  }
  llvm_unreachable("Unhandled ArgKind");
}

VariantMatcher::Payload::~Payload() = default;

class VariantMatcher::SinglePayload : public VariantMatcher::Payload {
public:
  explicit SinglePayload(DynMatcher matcher) : matcher(std::move(matcher)) {}

  std::optional<DynMatcher> getDynMatcher() const override { return matcher; }

  std::string getTypeAsString() const override { return "Matcher"; }

private:
  DynMatcher matcher;
};

VariantMatcher::VariantMatcher() = default;

VariantMatcher VariantMatcher::SingleMatcher(DynMatcher matcher) {
  return VariantMatcher(std::make_shared<SinglePayload>(std::move(matcher)));
}

std::optional<DynMatcher> VariantMatcher::getDynMatcher() const {
  return value ? value->getDynMatcher() : std::nullopt;
}

void VariantMatcher::reset() { value.reset(); }

std::string VariantMatcher::getTypeAsString() const { return "<Nothing>"; }

VariantValue::VariantValue(const VariantValue &other) : type(VT_Nothing) {
  *this = other;
}

VariantValue::VariantValue(const StringRef string) : type(VT_String) {
  value.String = new StringRef(string);
}

VariantValue::VariantValue(const VariantMatcher &matcher) : type(VT_Matcher) {
  value.Matcher = new VariantMatcher(matcher);
}

VariantValue::~VariantValue() { reset(); }

VariantValue &VariantValue::operator=(const VariantValue &other) {
  if (this == &other)
    return *this;
  reset();
  switch (other.type) {
  case VT_String:
    setString(other.getString());
    break;
  case VT_Matcher:
    setMatcher(other.getMatcher());
    break;
  case VT_Nothing:
    type = VT_Nothing;
    break;
  }
  return *this;
}

void VariantValue::reset() {
  switch (type) {
  case VT_String:
    delete value.String;
    break;
  case VT_Matcher:
    delete value.Matcher;
    break;
  // Cases that do nothing.
  case VT_Nothing:
    break;
  }
  type = VT_Nothing;
}

bool VariantValue::isString() const { return type == VT_String; }

const StringRef &VariantValue::getString() const {
  assert(isString());
  return *value.String;
}

void VariantValue::setString(const StringRef &newValue) {
  reset();
  type = VT_String;
  value.String = new StringRef(newValue);
}

bool VariantValue::isMatcher() const { return type == VT_Matcher; }

const VariantMatcher &VariantValue::getMatcher() const {
  assert(isMatcher());
  return *value.Matcher;
}

void VariantValue::setMatcher(const VariantMatcher &newValue) {
  reset();
  type = VT_Matcher;
  value.Matcher = new VariantMatcher(newValue);
}

std::string VariantValue::getTypeAsString() const {
  switch (type) {
  case VT_String:
    return "String";
  case VT_Matcher:
    return "Matcher";
  case VT_Nothing:
    return "Nothing";
  }
  llvm_unreachable("Invalid Type");
}

} // namespace mlir::query::matcher
