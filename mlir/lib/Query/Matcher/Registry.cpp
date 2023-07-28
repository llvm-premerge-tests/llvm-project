//===- MatcherRegistry.cpp - Matcher registry -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registry map populated at static initialization time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/Registry.h"

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"
#include <set>
#include <utility>

namespace mlir::query::matcher {
namespace {

using ConstructorMap =
    llvm::StringMap<std::unique_ptr<const internal::MatcherDescriptor>>;

// This is needed because these matchers are defined as overloaded functions.
using IsConstantOp = detail::constant_op_matcher();
using HasOpAttrName = detail::AttrOpMatcher(StringRef);
using HasOpName = detail::NameOpMatcher(StringRef);

class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return constructorMap; }

private:
  void registerMatcher(llvm::StringRef matcherName,
                       std::unique_ptr<internal::MatcherDescriptor> callback);

  ConstructorMap constructorMap;
};

} // namespace

void RegistryMaps::registerMatcher(
    llvm::StringRef matcherName,
    std::unique_ptr<internal::MatcherDescriptor> callback) {
  assert(!constructorMap.contains(matcherName));
  constructorMap[matcherName] = std::move(callback);
}

// Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  auto registerOpMatcher = [&](const std::string &name, auto matcher) {
    registerMatcher(name, internal::makeMatcherAutoMarshall(matcher, name));
  };

  // Register matchers using the template function (added in alphabetical order
  // for consistency)
  registerOpMatcher("hasOpAttrName", static_cast<HasOpAttrName *>(m_Attr));
  registerOpMatcher("hasOpName", static_cast<HasOpName *>(m_Op));
  registerOpMatcher("isConstantOp", static_cast<IsConstantOp *>(m_Constant));
  registerOpMatcher("isNegInfFloat", m_NegInfFloat);
  registerOpMatcher("isNegZeroFloat", m_NegZeroFloat);
  registerOpMatcher("isNonZero", m_NonZero);
  registerOpMatcher("isOne", m_One);
  registerOpMatcher("isOneFloat", m_OneFloat);
  registerOpMatcher("isPosInfFloat", m_PosInfFloat);
  registerOpMatcher("isPosZeroFloat", m_PosZeroFloat);
  registerOpMatcher("isZero", m_Zero);
  registerOpMatcher("isZeroFloat", m_AnyZeroFloat);
}

RegistryMaps::~RegistryMaps() = default;

static llvm::ManagedStatic<RegistryMaps> registryData;

std::optional<MatcherCtor>
Registry::lookupMatcherCtor(llvm::StringRef matcherName) {
  auto it = registryData->constructors().find(matcherName);
  return it == registryData->constructors().end() ? std::optional<MatcherCtor>()
                                                  : it->second.get();
}

std::vector<ArgKind> Registry::getAcceptedCompletionTypes(
    ArrayRef<std::pair<MatcherCtor, unsigned>> context) {
  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ArgKind> typeSet;
  typeSet.insert(ArgKind(ArgKind::AK_Matcher));

  for (const auto &ctxEntry : context) {
    MatcherCtor ctor = ctxEntry.first;
    unsigned argNumber = ctxEntry.second;
    std::vector<ArgKind> nextTypeSet;

    if (argNumber < ctor->getNumArgs())
      ctor->getArgKinds(argNumber, nextTypeSet);

    typeSet.insert(nextTypeSet.begin(), nextTypeSet.end());
  }

  return std::vector<ArgKind>(typeSet.begin(), typeSet.end());
}

std::vector<MatcherCompletion>
Registry::getMatcherCompletions(ArrayRef<ArgKind> acceptedTypes) {
  std::vector<MatcherCompletion> completions;

  // Search the registry for acceptable matchers.
  for (const auto &m : registryData->constructors()) {
    const internal::MatcherDescriptor &matcher = *m.getValue();
    StringRef name = m.getKey();

    unsigned numArgs = matcher.getNumArgs();
    std::vector<std::vector<ArgKind>> argKinds(numArgs);

    for (const ArgKind &kind : acceptedTypes) {
      if (kind.getArgKind() != kind.AK_Matcher)
        continue;

      for (unsigned arg = 0; arg != numArgs; ++arg)
        matcher.getArgKinds(arg, argKinds[arg]);
    }

    std::string decl;
    llvm::raw_string_ostream OS(decl);

    std::string typedText = std::string(name);
    OS << "Matcher: " << name << "(";

    for (const std::vector<ArgKind> &arg : argKinds) {
      if (&arg != &argKinds[0])
        OS << ", ";

      bool firstArgKind = true;
      // Two steps. First all non-matchers, then matchers only.
      for (const ArgKind &argKind : arg) {
        if (!firstArgKind)
          OS << "|";

        firstArgKind = false;
        OS << argKind.asString();
      }
    }

    OS << ")";
    typedText += "(";

    if (argKinds.empty())
      typedText += ")";
    else if (argKinds[0][0].getArgKind() == ArgKind::AK_String)
      typedText += "\"";

    completions.emplace_back(typedText, OS.str());
  }

  return completions;
}

VariantMatcher Registry::constructMatcher(MatcherCtor ctor,
                                          SourceRange nameRange,
                                          ArrayRef<ParserValue> args,
                                          Diagnostics *error) {
  return ctor->create(nameRange, args, error);
}

} // namespace mlir::query::matcher
