//===- llvm/Support/UnicodeNameToCodepoint.cpp - Unicode character properties
//-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements functions to map the name or alias of a unicode
// character to its codepoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Unicode.h"

namespace llvm {
namespace sys {
namespace unicode {

extern const char *UnicodeNameToCodepointDict;
extern const uint8_t *UnicodeNameToCodepointIndex;
extern const std::size_t UnicodeNameToCodepointIndexSize;

using BufferType = SmallString<64>;

struct node {
  char32_t value = 0xFFFFFFFF;
  uint32_t children_offset = 0;
  bool has_sibling = false;
  uint32_t size = 0;
  StringRef name;

  constexpr bool is_valid() const {
    return name.size() != 0 || value == 0xFFFFFFFF;
  }
  constexpr bool has_children() const { return children_offset != 0; }
};

static node read_node(uint32_t offset) {
  const uint32_t origin = offset;
  node n;

  uint8_t nameInfo = UnicodeNameToCodepointIndex[offset++];
  if (offset + 6 >= UnicodeNameToCodepointIndexSize)
    return n;

  const bool long_name = nameInfo & 0x40;
  const bool has_value = nameInfo & 0x80;
  std::size_t size = nameInfo & ~0xC0;
  if (long_name) {
    uint32_t name_offset = (UnicodeNameToCodepointIndex[offset++] << 8);
    name_offset |= UnicodeNameToCodepointIndex[offset++];
    n.name = StringRef(UnicodeNameToCodepointDict + name_offset, size);
  } else {
    n.name = StringRef(UnicodeNameToCodepointDict + size, 1);
  }
  if (has_value) {
    uint8_t h = UnicodeNameToCodepointIndex[offset++];
    uint8_t m = UnicodeNameToCodepointIndex[offset++];
    uint8_t l = UnicodeNameToCodepointIndex[offset++];
    n.value = ((h << 16) | (m << 8) | l) >> 3;

    bool has_children = l & 0x02;
    n.has_sibling = l & 0x01;

    if (has_children) {
      n.children_offset = UnicodeNameToCodepointIndex[offset++] << 16;
      n.children_offset |= UnicodeNameToCodepointIndex[offset++] << 8;
      n.children_offset |= UnicodeNameToCodepointIndex[offset++];
    }
  } else {
    uint8_t h = UnicodeNameToCodepointIndex[offset++];
    n.has_sibling = h & 0x80;
    bool has_children = h & 0x40;
    h &= ~0xC0;
    if (has_children) {
      n.children_offset = (h << 16);
      n.children_offset |=
          (uint32_t(UnicodeNameToCodepointIndex[offset++]) << 8);
      n.children_offset |= UnicodeNameToCodepointIndex[offset++];
    }
  }
  n.size = offset - origin;
  return n;
}

static bool StartsWith(StringRef Name, StringRef Needle, bool Strict,
                       std::size_t &Consummed, char &PreviousCharInName,
                       char &PreviousCharInNeedle) {

  Consummed = 0;
  if (Strict) {
    if (!Name.startswith(Needle))
      return false;
    Consummed = Needle.size();
    return true;
  }
  if (Needle.size() == 0)
    return true;

  auto NamePos = Name.begin();
  auto NeedlePos = Needle.begin();

  char PreviousCharInNameOrigin = PreviousCharInName;
  char PreviousCharInNeedleOrigin = PreviousCharInNeedle;

  auto IgnoreSpaces = [](auto It, auto End, char &PreviousChar) {
    while (It != End) {
      bool Ignore =
          *It == ' ' || *It == '_' || (*It == '-' && PreviousChar != ' ');
      PreviousChar = *It;
      if (!Ignore)
        break;
      ++It;
    }
    return It;
  };

  while (true) {
    NeedlePos = IgnoreSpaces(NeedlePos, Needle.end(), PreviousCharInNeedle);
    NamePos = IgnoreSpaces(NamePos, Name.end(), PreviousCharInName);
    if (NeedlePos == Needle.end())
      break;
    if (NamePos == Name.end())
      break;
    if (toUpper(*NeedlePos) != toUpper(*NamePos))
      break;
    NeedlePos++;
    NamePos++;
  }
  Consummed = std::distance(Name.begin(), NamePos);
  if (NeedlePos == Needle.end()) {
    PreviousCharInName = PreviousCharInNameOrigin;
    PreviousCharInNeedle = PreviousCharInNeedleOrigin;
  }
  return NeedlePos == Needle.end();
}

static std::tuple<node, bool, uint32_t>
compare_node(uint32_t offset, StringRef name, bool Strict,
             char PreviousCharInName, char PreviousCharInNeedle,
             BufferType &Buffer) {
  auto n = read_node(offset);
  std::size_t Consummed = 0;
  bool DoesStartWith = StartsWith(name, n.name, Strict, Consummed,
                                  PreviousCharInName, PreviousCharInNeedle);
  if (!DoesStartWith) {
    return {n, false, 0};
  }

  if (name.size() - Consummed == 0) {
    // std::reverse_copy(n.name.begin(), n.name.end(),
    // std::back_inserter(Buffer));
    return {n, true, n.value};
  }
  if (n.has_children()) {
    auto o = n.children_offset;
    for (;;) {
      node c;
      bool matches;
      uint32_t value;
      std::tie(c, matches, value) =
          compare_node(o, name.substr(Consummed), Strict, PreviousCharInName,
                       PreviousCharInNeedle, Buffer);
      if (matches) {
        std::reverse_copy(c.name.begin(), c.name.end(),
                          std::back_inserter(Buffer));
        return {n, true, value};
      }
      o += c.size;
      if (!c.has_sibling)
        break;
    }
  }
  return {n, false, 0};
}

static std::tuple<node, bool, uint32_t>
compare_node(uint32_t offset, StringRef Name, bool Strict, BufferType &Buffer) {
  return compare_node(offset, Name, Strict, 0, 0, Buffer);
}

// clang-format off
constexpr const char *const hangul_syllables[][3] = {
    { "G",  "A",   ""   },
    { "GG", "AE",  "G"  },
    { "N",  "YA",  "GG" },
    { "D",  "YAE", "GS" },
    { "DD", "EO",  "N", },
    { "R",  "E",   "NJ" },
    { "M",  "YEO", "NH" },
    { "B",  "YE",  "D"  },
    { "BB", "O",   "L"  },
    { "S",  "WA",  "LG" },
    { "SS", "WAE", "LM" },
    { "",   "OE",  "LB" },
    { "J",  "YO",  "LS" },
    { "JJ", "U",   "LT" },
    { "C",  "WEO", "LP" },
    { "K",  "WE",  "LH" },
    { "T",  "WI",  "M"  },
    { "P",  "YU",  "B"  },
    { "H",  "EU",  "BS" },
    { 0,    "YI",  "S"  },
    { 0,    "I",   "SS" },
    { 0,    0,     "NG" },
    { 0,    0,     "J"  },
    { 0,    0,     "C"  },
    { 0,    0,     "K"  },
    { 0,    0,     "T"  },
    { 0,    0,     "P"  },
    { 0,    0,     "H"  }
    };
// clang-format on

// Unicode 14.0  3.12 Conjoining Jamo Behavior
//  Common constants
constexpr const char32_t SBase = 0xAC00;
// constexpr const char32_t LBase = 0x1100;
// constexpr const char32_t VBase = 0x1161;
// constexpr const char32_t TBase = 0x11A7;
constexpr const uint32_t LCount = 19;
constexpr const uint32_t VCount = 21;
constexpr const uint32_t TCount = 28;
// constexpr const uint32_t NCount = (VCount * TCount);
// constexpr const uint32_t SCount = (LCount * NCount);

static std::size_t findSyllable(StringRef Name, bool Strict,
                                char &PreviousInName, int &Pos, int Column) {
  assert(Column == 0 || Column == 1 || Column == 2);
  static std::size_t CountPerColumn[] = {LCount, VCount, TCount};
  char NeedleStart = 0;
  int Len = -1;
  int Prev = PreviousInName;
  for (std::size_t i = 0; i < CountPerColumn[Column]; i++) {
    StringRef Syllable(hangul_syllables[i][Column]);
    if (int(Syllable.size()) <= Len)
      continue;
    std::size_t Consummed = 0;
    char PreviousInNameCopy = PreviousInName;
    bool DoesStartWith = StartsWith(Name, Syllable, Strict, Consummed,
                                    PreviousInNameCopy, NeedleStart);
    if (!DoesStartWith)
      continue;
    Len = Consummed;
    Pos = i;
    Prev = PreviousInNameCopy;
  }
  if (Len == -1)
    return 0;
  PreviousInName = Prev;
  return size_t(Len);
}

static llvm::Optional<char32_t>
nameToHangulCodePoint(StringRef Name, bool Strict, BufferType &Buffer) {
  Buffer.clear();
  // Hangul Syllable Decomposition
  std::size_t Consummed = 0;
  char NameStart = 0, NeedleStart = 0;
  bool DoesStartWith = StartsWith(Name, "HANGUL SYLLABLE ", Strict, Consummed,
                                  NameStart, NeedleStart);
  if (!DoesStartWith)
    return {};
  Name = Name.substr(Consummed);
  int L = -1, V = -1, T = -1;
  Name = Name.substr(findSyllable(Name, Strict, NameStart, L, 0));
  Name = Name.substr(findSyllable(Name, Strict, NameStart, V, 1));
  Name = Name.substr(findSyllable(Name, Strict, NameStart, T, 2));
  if (L != -1 && V != -1 && T != -1 && Name.size() == 0) {
    if (!Strict) {
      Buffer.append("HANGUL SYLLABLE ");
      if (L != -1)
        Buffer.append(hangul_syllables[L][0]);
      if (V != -1)
        Buffer.append(hangul_syllables[V][1]);
      if (T != -1)
        Buffer.append(hangul_syllables[T][2]);
    }
    return SBase + (std::uint32_t(L) * VCount + std::uint32_t(V)) * TCount +
           std::uint32_t(T);
  }
  // Otherwise, it's an illegal syllable name.
  return {};
}

struct generated_name_data {
  StringRef prefix;
  uint32_t start;
  uint32_t end;
};

// Unicode 14.0 Table 4-8. Name Derivation Rule Prefix Strings
// This needs to be kept in sync with
// llvm/utils/UnicodeData/UnicodeNameMappingGenerator.cpp
static const generated_name_data generated_name_data_table[] = {
    {"CJK UNIFIED IDEOGRAPH-", 0x3400, 0x4DBF},
    {"CJK UNIFIED IDEOGRAPH-", 0x4E00, 0x9FFC},
    {"CJK UNIFIED IDEOGRAPH-", 0x20000, 0x2A6DD},
    {"CJK UNIFIED IDEOGRAPH-", 0x2A700, 0x2B734},
    {"CJK UNIFIED IDEOGRAPH-", 0x2B740, 0x2B81D},
    {"CJK UNIFIED IDEOGRAPH-", 0x2B820, 0x2CEA1},
    {"CJK UNIFIED IDEOGRAPH-", 0x2CEB0, 0x2EBE0},
    {"CJK UNIFIED IDEOGRAPH-", 0x30000, 0x3134A},
    {"TANGUT IDEOGRAPH-", 0x17000, 0x187F7},
    {"TANGUT IDEOGRAPH-", 0x18D00, 0x18D08},
    {"KHITAN SMALL SCRIPT CHARACTER-", 0x18B00, 0x18CD5},
    {"NUSHU CHARACTER-", 0x1B170, 0x1B2FB},
    {"CJK COMPATIBILITY IDEOGRAPH-", 0xF900, 0xFA6D},
    {"CJK COMPATIBILITY IDEOGRAPH-", 0xFA70, 0xFAD9},
    {"CJK COMPATIBILITY IDEOGRAPH-", 0x2F800, 0x2FA1D},
};

static llvm::Optional<char32_t>
nameToGeneratedCodePoint(StringRef Name, bool Strict, BufferType &Buffer) {
  for (auto &&item : generated_name_data_table) {
    Buffer.clear();
    std::size_t Consummed = 0;
    char NameStart = 0, NeedleStart = 0;
    bool DoesStartWith = StartsWith(Name, item.prefix, Strict, Consummed,
                                    NameStart, NeedleStart);
    if (!DoesStartWith)
      continue;
    auto Number = Name.substr(Consummed);
    unsigned long long v = 0;
    // Be consistent about mandating uper casing
    if (Strict &&
        llvm::any_of(Number, [](char c) { return c >= 'a' && c <= 'f'; }))
      return {};
    if (getAsUnsignedInteger(Number, 16, v) || v < item.start || v > item.end)
      continue;
    if (!Strict) {
      Buffer.append(item.prefix);
      Buffer.append(to_hexString(v, true));
    }
    return v;
  }
  return {};
}

static llvm::Optional<char32_t> nameToCodepoint(StringRef name, bool Strict,
                                                BufferType &Buffer) {

  if (name.empty())
    return {};
  llvm::Optional<char32_t> Res = nameToHangulCodePoint(name, Strict, Buffer);
  if (!Res)
    Res = nameToGeneratedCodePoint(name, Strict, Buffer);
  if (Res)
    return *Res;

  uint32_t offset = 0;
  for (;;) {
    Buffer.clear();
    node node;
    bool matches;
    uint32_t value;
    std::tie(node, matches, value) = compare_node(offset, name, Strict, Buffer);
    if (!node.is_valid())
      continue;
    if (matches) {
      std::reverse_copy(node.name.begin(), node.name.end(),
                        std::back_inserter(Buffer));
      // UAX44-LM2. Ignore case, whitespace, underscore ('_'), and all medial
      // hyphens except the hyphen in U+1180 HANGUL JUNGSEONG O-E.
      if (Strict && value == 0x116c &&
          name.find_insensitive("O-E") != StringRef::npos) {
        Buffer = "HANGUL JUNGSEONG O-E";
        value = 0x1180;
      }
      return value;
    }
    if (!node.has_sibling)
      break;
    offset += node.size;
  }
  return {};
}

llvm::Optional<char32_t> nameToCodepointStrict(StringRef name) {

  BufferType Buffer;
  auto Opt = nameToCodepoint(name, true, Buffer);
  return Opt;
}

llvm::Optional<LooseMatchingResult>
nameToCodepointLooseMatching(StringRef name) {
  BufferType Buffer;
  auto Opt = nameToCodepoint(name, false, Buffer);
  if (!Opt)
    return {};
  std::reverse(Buffer.begin(), Buffer.end());
  return LooseMatchingResult{*Opt, Buffer};
}

} // namespace unicode

} // namespace sys
} // namespace llvm
