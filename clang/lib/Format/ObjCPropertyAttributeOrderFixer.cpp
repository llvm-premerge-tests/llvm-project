//===--- ObjCPropertyAttributeOrderFixer.cpp -------------------*- C++--*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ObjCPropertyAttributeOrderFixer, a TokenAnalyzer that
/// adjusts the order of attributes in an ObjC `@property(...)` declaration,
/// depending on the style.
///
//===----------------------------------------------------------------------===//

#include "ObjCPropertyAttributeOrderFixer.h"

#include "FormatToken.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "format-objc-property-attribute-order-fixer"

namespace clang {
namespace format {

ObjCPropertyAttributeOrderFixer::ObjCPropertyAttributeOrderFixer(
    const Environment &Env, const FormatStyle &Style)
    : TokenAnalyzer(Env, Style) {

  // Create an "order priority" map to use to sort properties.
  unsigned index = 0;
  for (auto const &Property : Style.ObjCPropertyAttributeOrder)
    SortOrderMap[Property] = index++;
  // A sentinel value bigger than all others (used to sort unknown ones to the
  // end).
  SortOrderMax = index;
}

struct ObjCPropertyEntry {
  StringRef Attribute; // eg, "readwrite"
  StringRef Value;     // eg, the "foo" of the attribute "getter=foo"
};

static bool isObjCPropertyAttribute(const FormatToken *Tok) {
  // Most attributes look like identifiers, but `class` is a keyword.
  return Tok->isOneOf(tok::identifier, tok::kw_class);
}

void ObjCPropertyAttributeOrderFixer::sortPropertyAttributes(
    const SourceManager &SourceMgr, tooling::Replacements &Fixes,
    const FormatToken *LParenTok, const FormatToken *RParenTok) const {
  // Skip past any leading comments.
  const FormatToken *const BeginTok = LParenTok->getNextNonComment();

  // Block out any trailing comments. ("End" marks a left-closed interval, so
  // store one-past-last) This will point to either the right-paren, or a
  // comment (if there were multiple trailing comments).
  const FormatToken *const EndTok = RParenTok->getPreviousNonComment()->Next;
  assert(EndTok->isOneOf(tok::r_paren, tok::comment) &&
         "Expect the range to be bounded by comment or paren");

  // If there are zero or one elements, nothing to do.
  if (BeginTok == EndTok || BeginTok->Next == EndTok)
    return;

  // Collect the attributes.
  SmallVector<ObjCPropertyEntry, 8> PropertyAttributes;
  for (auto Tok = BeginTok; Tok != EndTok; Tok = Tok->Next) {
    if (Tok->is(tok::comma)) {
      // Ignore the comma separators.
      continue;
    } else if (isObjCPropertyAttribute(Tok)) {
      // Memoize the attribute. (Note that 'class' is a legal attribute!)
      PropertyAttributes.push_back({Tok->TokenText.trim(), StringRef{}});

      // Also handle `getter=getFoo` attributes.
      // (Note: no check needed against `EndTok`, since its type is not
      // BinaryOperator or Identifier)
      if (Tok->Next->is(tok::equal)) {
        Tok = Tok->Next;
        if (Tok->Next->is(tok::identifier)) {
          Tok = Tok->Next;
          PropertyAttributes.back().Value = Tok->TokenText.trim();
        } else {
          // If we hit any other kind of token, just bail. It's unusual/illegal.
          return;
        }
      }
    } else {
      // If we hit any other kind of token, just bail.
      return;
    }
  }

  // Create a "remapping index" on how to reorder the attributes.
  SmallVector<unsigned, 8> Indices =
      llvm::to_vector<8>(llvm::seq<unsigned>(0, PropertyAttributes.size()));

  // Sort the indices based on the priority stored in 'SortOrderMap'; use Max
  // for missing values.
  auto sortIndex = [&](const StringRef &needle) -> unsigned {
    auto i = SortOrderMap.find(needle);
    return (i == SortOrderMap.end()) ? SortOrderMax : i->getValue();
  };
  llvm::stable_sort(Indices, [&](unsigned LHSI, unsigned RHSI) {
    return sortIndex(PropertyAttributes[LHSI].Attribute) <
           sortIndex(PropertyAttributes[RHSI].Attribute);
  });

  // Deduplicate the attributes.
  Indices.erase(std::unique(Indices.begin(), Indices.end(),
                            [&](unsigned LHSI, unsigned RHSI) {
                              return PropertyAttributes[LHSI].Attribute ==
                                     PropertyAttributes[RHSI].Attribute;
                            }),
                Indices.end());

  // If there are no removals or shuffling, then don't suggest any fixup.
  if (Indices.size() == PropertyAttributes.size() && llvm::is_sorted(Indices))
    return;

  // Generate the replacement text.
  std::string NewText;
  for (unsigned Index : Indices) {
    if (!NewText.empty())
      NewText += ", ";

    NewText += PropertyAttributes[Index].Attribute;

    if (!PropertyAttributes[Index].Value.empty()) {
      NewText += "=";
      NewText += PropertyAttributes[Index].Value;
    }
  }

  auto Range = CharSourceRange::getCharRange(
      BeginTok->getStartOfNonWhitespace(), EndTok->Previous->Tok.getEndLoc());
  auto Replacement = tooling::Replacement(SourceMgr, Range, NewText);
  auto Err = Fixes.add(Replacement);
  if (Err) {
    llvm::errs() << "Error while reodering ObjC property attributes : "
                 << llvm::toString(std::move(Err)) << "\n";
  }
}

const FormatToken *ObjCPropertyAttributeOrderFixer::analyzeObjCPropertyDecl(
    const SourceManager &SourceMgr, const AdditionalKeywords &Keywords,
    tooling::Replacements &Fixes, const FormatToken *const Tok) const {
  // Expect `property` to be the very next token or else just bail early.
  const FormatToken *const PropertyTok = Tok->Next;
  if (!PropertyTok || PropertyTok->TokenText != "property")
    return Tok;

  // Expect the opening paren to be the next token or else just bail early.
  const FormatToken *const LParenTok = PropertyTok->getNextNonComment();
  if (!LParenTok || LParenTok->isNot(tok::l_paren))
    return Tok;

  // Get the matching right-paren, the bounds for property attributes.
  const FormatToken *const RParenTok = LParenTok->MatchingParen;
  if (!RParenTok)
    return Tok;

  sortPropertyAttributes(SourceMgr, Fixes, LParenTok, RParenTok);

  // Return the final token since we can skip past everything in between.
  return RParenTok;
}

std::pair<tooling::Replacements, unsigned>
ObjCPropertyAttributeOrderFixer::analyze(
    TokenAnnotator & /*Annotator*/,
    SmallVectorImpl<AnnotatedLine *> &AnnotatedLines,
    FormatTokenLexer &Tokens) {
  tooling::Replacements Fixes;
  const AdditionalKeywords &Keywords = Tokens.getKeywords();
  const SourceManager &SourceMgr = Env.getSourceManager();
  AffectedRangeMgr.computeAffectedLines(AnnotatedLines);

  for (AnnotatedLine *Line : AnnotatedLines) {
    if (!Line->Affected || Line->InPPDirective)
      continue;
    FormatToken *First = Line->First;
    assert(First);
    if (First->Finalized)
      continue;

    const auto *Last = Line->Last;

    for (const auto *Tok = First; Tok && Tok != Last && Tok->Next;
         Tok = Tok->Next) {
      // Skip until the `@` of a `@property` declaration.
      if (Tok->isNot(TT_ObjCProperty))
        continue;
      Tok = analyzeObjCPropertyDecl(SourceMgr, Keywords, Fixes, Tok);
    }
  }
  return {Fixes, 0};
}

} // namespace format
} // namespace clang
