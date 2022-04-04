//===- unittests/Support/UnicodeTest.cpp - Unicode.h tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Unicode.h"
#include "llvm/Support/ConvertUTF.h"
#include "gtest/gtest.h"

namespace llvm {
namespace sys {
namespace unicode {
namespace {

TEST(Unicode, columnWidthUTF8) {
  EXPECT_EQ(0, columnWidthUTF8(""));
  EXPECT_EQ(1, columnWidthUTF8(" "));
  EXPECT_EQ(1, columnWidthUTF8("a"));
  EXPECT_EQ(1, columnWidthUTF8("~"));

  EXPECT_EQ(6, columnWidthUTF8("abcdef"));

  EXPECT_EQ(-1, columnWidthUTF8("\x01"));
  EXPECT_EQ(-1, columnWidthUTF8("\t"));
  EXPECT_EQ(-1, columnWidthUTF8("aaaaaaaaaa\x01"));
  EXPECT_EQ(-1, columnWidthUTF8("\342\200\213")); // 200B ZERO WIDTH SPACE

  // 00AD SOFT HYPHEN is displayed on most terminals as a space or a dash. Some
  // text editors display it only when a line is broken at it, some use it as a
  // line-break hint, but don't display. We choose terminal-oriented
  // interpretation.
  EXPECT_EQ(1, columnWidthUTF8("\302\255"));

  EXPECT_EQ(0, columnWidthUTF8("\314\200"));     // 0300 COMBINING GRAVE ACCENT
  EXPECT_EQ(1, columnWidthUTF8("\340\270\201")); // 0E01 THAI CHARACTER KO KAI
  EXPECT_EQ(2, columnWidthUTF8("\344\270\200")); // CJK UNIFIED IDEOGRAPH-4E00

  EXPECT_EQ(4, columnWidthUTF8("\344\270\200\344\270\200"));
  EXPECT_EQ(3, columnWidthUTF8("q\344\270\200"));
  EXPECT_EQ(3, columnWidthUTF8("\314\200\340\270\201\344\270\200"));

  // Invalid UTF-8 strings, columnWidthUTF8 should error out.
  EXPECT_EQ(-2, columnWidthUTF8("\344"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270\033"));
  EXPECT_EQ(-2, columnWidthUTF8("\344\270\300"));
  EXPECT_EQ(-2, columnWidthUTF8("\377\366\355"));

  EXPECT_EQ(-2, columnWidthUTF8("qwer\344"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270\033"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\344\270\300"));
  EXPECT_EQ(-2, columnWidthUTF8("qwer\377\366\355"));

  // UTF-8 sequences longer than 4 bytes correspond to unallocated Unicode
  // characters.
  EXPECT_EQ(-2, columnWidthUTF8("\370\200\200\200\200"));     // U+200000
  EXPECT_EQ(-2, columnWidthUTF8("\374\200\200\200\200\200")); // U+4000000
}

TEST(Unicode, isPrintable) {
  EXPECT_FALSE(isPrintable(0)); // <control-0000>-<control-001F>
  EXPECT_FALSE(isPrintable(0x01));
  EXPECT_FALSE(isPrintable(0x1F));
  EXPECT_TRUE(isPrintable(' '));
  EXPECT_TRUE(isPrintable('A'));
  EXPECT_TRUE(isPrintable('~'));
  EXPECT_FALSE(isPrintable(0x7F)); // <control-007F>..<control-009F>
  EXPECT_FALSE(isPrintable(0x90));
  EXPECT_FALSE(isPrintable(0x9F));

  EXPECT_TRUE(isPrintable(0xAC));
  EXPECT_TRUE(isPrintable(0xAD)); // SOFT HYPHEN is displayed on most terminals
                                  // as either a space or a dash.
  EXPECT_TRUE(isPrintable(0xAE));

  EXPECT_TRUE(isPrintable(0x0377));  // GREEK SMALL LETTER PAMPHYLIAN DIGAMMA
  EXPECT_FALSE(isPrintable(0x0378)); // <reserved-0378>..<reserved-0379>

  EXPECT_FALSE(isPrintable(0x0600)); // ARABIC NUMBER SIGN

  EXPECT_FALSE(isPrintable(0x1FFFF)); // <reserved-1F774>..<noncharacter-1FFFF>
  EXPECT_TRUE(isPrintable(0x20000));  // CJK UNIFIED IDEOGRAPH-20000

  EXPECT_FALSE(isPrintable(0x10FFFF)); // noncharacter

  // test the validity of a fast path in columnWidthUTF8
  for (unsigned char c = 0; c < 128; ++c) {
    const UTF8 buf8[2] = {c, 0};
    const UTF8 *Target8 = &buf8[0];
    UTF32 buf32[1];
    UTF32 *Target32 = &buf32[0];
    auto status = ConvertUTF8toUTF32(&Target8, Target8 + 1, &Target32,
                                     Target32 + 1, strictConversion);
    EXPECT_EQ(status, conversionOK);
    EXPECT_EQ((columnWidthUTF8(reinterpret_cast<const char *>(buf8)) == 1),
              (bool)isPrintable(buf32[0]));
  }
}

/*TEST(Unicode, nameToCodepointStrict) {
    auto map = [](StringRef Str) {
        return nameToCodepointStrict(Str).getValueOr(0xFFFF'FFFF);
    };

  // generated codepoints
  EXPECT_EQ(0x03400, map("CJK UNIFIED IDEOGRAPH-3400"));
  EXPECT_EQ(0x04DBF, map("CJK UNIFIED IDEOGRAPH-4DBF"));
  EXPECT_EQ(0x04E00, map("CJK UNIFIED IDEOGRAPH-4E00"));
  EXPECT_EQ(0x09FFC, map("CJK UNIFIED IDEOGRAPH-9FFC"));
  EXPECT_EQ(0x20000, map("CJK UNIFIED IDEOGRAPH-20000"));
  EXPECT_EQ(0x2A6DD, map("CJK UNIFIED IDEOGRAPH-2A6DD"));
  EXPECT_EQ(0x2A700, map("CJK UNIFIED IDEOGRAPH-2A700"));
  EXPECT_EQ(0x2B740, map("CJK UNIFIED IDEOGRAPH-2B740"));
  EXPECT_EQ(0x2B81D, map("CJK UNIFIED IDEOGRAPH-2B81D"));
  EXPECT_EQ(0x2B820, map("CJK UNIFIED IDEOGRAPH-2B820"));
  EXPECT_EQ(0x2CEA1, map("CJK UNIFIED IDEOGRAPH-2CEA1"));
  EXPECT_EQ(0x2CEB0, map("CJK UNIFIED IDEOGRAPH-2CEB0"));
  EXPECT_EQ(0x2EBE0, map("CJK UNIFIED IDEOGRAPH-2EBE0"));
  EXPECT_EQ(0x30000, map("CJK UNIFIED IDEOGRAPH-30000"));
  EXPECT_EQ(0x3134A, map("CJK UNIFIED IDEOGRAPH-3134A"));
  EXPECT_EQ(0x17000, map("TANGUT IDEOGRAPH-17000"));
  EXPECT_EQ(0x187F7, map("TANGUT IDEOGRAPH-187F7"));
  EXPECT_EQ(0x18D00, map("TANGUT IDEOGRAPH-18D00"));
  EXPECT_EQ(0x18D08, map("TANGUT IDEOGRAPH-18D08"));
  EXPECT_EQ(0x18B00, map("KHITAN SMALL SCRIPT CHARACTER-18B00"));
  EXPECT_EQ(0x18CD5, map("KHITAN SMALL SCRIPT CHARACTER-18CD5"));
  EXPECT_EQ(0x1B170, map("NUSHU CHARACTER-1B170"));
  EXPECT_EQ(0x1B2FB, map("NUSHU CHARACTER-1B2FB"));
  EXPECT_EQ(0x0F900, map("CJK COMPATIBILITY IDEOGRAPH-F900"));
  EXPECT_EQ(0x0FA6D, map("CJK COMPATIBILITY IDEOGRAPH-FA6D"));
  EXPECT_EQ(0x0FA70, map("CJK COMPATIBILITY IDEOGRAPH-FA70"));
  EXPECT_EQ(0x0FAD9, map("CJK COMPATIBILITY IDEOGRAPH-FAD9"));
  EXPECT_EQ(0x2F800, map("CJK COMPATIBILITY IDEOGRAPH-2F800"));
  EXPECT_EQ(0x2FA1D, map("CJK COMPATIBILITY IDEOGRAPH-2FA1D"));

  EXPECT_EQ(0xAC00, map("HANGUL SYLLABLE GA"));
  EXPECT_EQ(0xAC14, map("HANGUL SYLLABLE GASS"));
  EXPECT_EQ(0xAC2B, map("HANGUL SYLLABLE GAELH"));
  EXPECT_EQ(0xAC7B, map("HANGUL SYLLABLE GEOLB"));
  EXPECT_EQ(0xC640, map("HANGUL SYLLABLE WA"));
  EXPECT_EQ(0xC544, map("HANGUL SYLLABLE A"));
  EXPECT_EQ(0xC5D0, map("HANGUL SYLLABLE E"));
  EXPECT_EQ(0xC774, map("HANGUL SYLLABLE I"));

  EXPECT_EQ(0x1F984, map("UNICORN FACE"));
  EXPECT_EQ(0x00640, map("ARABIC TATWEEL"));
  EXPECT_EQ(0x02C05, map("GLAGOLITIC CAPITAL LETTER YESTU"));
  EXPECT_EQ(0x13000, map("EGYPTIAN HIEROGLYPH A001"));
  EXPECT_EQ(0x02235, map("BECAUSE"));
  EXPECT_EQ(0x1F514, map("BELL"));
  EXPECT_EQ(0x1F9A9, map("FLAMINGO"));
  EXPECT_EQ(0x1F402, map("OX")); // 2 characters
  EXPECT_EQ(0xFBF9,
            map("ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                            "ABOVE WITH ALEF MAKSURA ISOLATED FORM"));

  // Aliases
  EXPECT_EQ(0x0000, map("NULL"));
  EXPECT_EQ(0x0007, map("ALERT"));
  EXPECT_EQ(0x0009, map("HORIZONTAL TABULATION"));
  EXPECT_EQ(0x0009, map("CHARACTER TABULATION"));
  EXPECT_EQ(0x000A, map("LINE FEED"));
  EXPECT_EQ(0x000A, map("NEW LINE"));
  EXPECT_EQ(0x0089, map("CHARACTER TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x0089,
            map("HORIZONTAL TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x2118,
            map("WEIERSTRASS ELLIPTIC FUNCTION"));     // correction
  EXPECT_EQ(0x2118, map("SCRIPT CAPITAL P"));          // correction
  EXPECT_EQ(0xFEFF, map("BYTE ORDER MARK"));           // alternate
  EXPECT_EQ(0xFEFF, map("ZERO WIDTH NO-BREAK SPACE")); // alternate

  // Should perform exact case match
  EXPECT_EQ(0xFFFFFFFF, map(""));
  EXPECT_EQ(0xFFFFFFFF, map("NOT A UNICODE CHARACTER"));
  EXPECT_EQ(0xFFFFFFFF, map("unicorn face"));
  EXPECT_EQ(0xFFFFFFFF, map("UNICORN FaCE"));
  EXPECT_EQ(0xFFFFFFFF, map("UNICORNFaCE"));
  EXPECT_EQ(0xFFFFFFFF, map("UNICORN"));
  EXPECT_EQ(0xFFFFFFFF, map("HANGUL SYLLABLE i"));
  EXPECT_EQ(0xFFFFFFFF, map("hANGUL SYLLABLE i"));
  EXPECT_EQ(0xFFFFFFFF, map("HANGULSYLLABLEI"));
  EXPECT_EQ(0xFFFFFFFF, map("HANGUL SYLLABLE"));
  EXPECT_EQ(0xFFFFFFFF, map("cJK COMPATIBILITY IDEOGRAPH-2FA1D"));
  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH-2FA1d"));
  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH 2FA1D"));
  EXPECT_EQ(0xFFFFFFFF,
            map("CJK COMPATIBILITY IDEOGRAPH-NOTANUMBER"));
  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH-1"));
  EXPECT_EQ(0xFFFFFFFF, map("ZERO WIDTH NO BREAK SPACE"));

  // Should not support abbreviations or figments
  EXPECT_EQ(0xFFFFFFFF, map("FVS1"));
  EXPECT_EQ(0xFFFFFFFF, map("HIGH OCTET PRESET"));
  EXPECT_EQ(0xFFFFFFFF, map("BEL"));
}*/

TEST(Unicode, nameToCodepointLoose) {
  auto map = [](StringRef Str) {
    auto Opt = nameToCodepointLooseMatching(Str);
    if (!Opt)
      return char32_t(0xFFFF'FFFF);
    return Opt->CodePoint;
  };

  // generated codepoints
  EXPECT_EQ(0x03400, map("CJK UNIFIED IDEOGRAPH-3400"));
  EXPECT_EQ(0x04DBF, map("CJK UNIFIED IDEOGRAPH-4DBF"));
  EXPECT_EQ(0x04E00, map("CJK UNIFIED IDEOGRAPH-4E00"));
  EXPECT_EQ(0x09FFC, map("CJK UNIFIED IDEOGRAPH-9FFC"));
  EXPECT_EQ(0x20000, map("CJK UNIFIED IDEOGRAPH-20000"));
  EXPECT_EQ(0x2A6DD, map("CJK UNIFIED IDEOGRAPH-2A6DD"));
  EXPECT_EQ(0x2A700, map("CJK UNIFIED IDEOGRAPH-2A700"));
  EXPECT_EQ(0x2B740, map("CJK UNIFIED IDEOGRAPH-2B740"));
  EXPECT_EQ(0x2B81D, map("CJK UNIFIED IDEOGRAPH-2B81D"));
  EXPECT_EQ(0x2B820, map("CJK UNIFIED IDEOGRAPH-2B820"));
  EXPECT_EQ(0x2CEA1, map("CJK UNIFIED IDEOGRAPH-2CEA1"));
  EXPECT_EQ(0x2CEB0, map("CJK UNIFIED IDEOGRAPH-2CEB0"));
  EXPECT_EQ(0x2EBE0, map("CJK UNIFIED IDEOGRAPH-2EBE0"));
  EXPECT_EQ(0x30000, map("CJK UNIFIED IDEOGRAPH-30000"));
  EXPECT_EQ(0x3134A, map("CJK UNIFIED IDEOGRAPH-3134A"));
  EXPECT_EQ(0x17000, map("TANGUT IDEOGRAPH-17000"));
  EXPECT_EQ(0x187F7, map("TANGUT IDEOGRAPH-187F7"));
  EXPECT_EQ(0x18D00, map("TANGUT IDEOGRAPH-18D00"));
  EXPECT_EQ(0x18D08, map("TANGUT IDEOGRAPH-18D08"));
  EXPECT_EQ(0x18B00, map("KHITAN SMALL SCRIPT CHARACTER-18B00"));
  EXPECT_EQ(0x18CD5, map("KHITAN SMALL SCRIPT CHARACTER-18CD5"));
  EXPECT_EQ(0x1B170, map("NUSHU CHARACTER-1B170"));
  EXPECT_EQ(0x1B2FB, map("NUSHU CHARACTER-1B2FB"));
  EXPECT_EQ(0x0F900, map("CJK COMPATIBILITY IDEOGRAPH-F900"));
  EXPECT_EQ(0x0FA6D, map("CJK COMPATIBILITY IDEOGRAPH-FA6D"));
  EXPECT_EQ(0x0FA70, map("CJK COMPATIBILITY IDEOGRAPH-FA70"));
  EXPECT_EQ(0x0FAD9, map("CJK COMPATIBILITY IDEOGRAPH-FAD9"));
  EXPECT_EQ(0x2F800, map("CJK COMPATIBILITY IDEOGRAPH-2F800"));
  EXPECT_EQ(0x2FA1D, map("CJK COMPATIBILITY IDEOGRAPH-2FA1D"));

  EXPECT_EQ(0xAC00, map("HANGUL SYLLABLE GA"));
  EXPECT_EQ(0xAC14, map("HANGUL SYLLABLE GASS"));
  EXPECT_EQ(0xAC2B, map("HANGUL SYLLABLE GAELH"));
  EXPECT_EQ(0xAC7B, map("HANGUL SYLLABLE GEOLB"));
  EXPECT_EQ(0xC640, map("HANGUL SYLLABLE WA"));
  EXPECT_EQ(0xC544, map("HANGUL SYLLABLE A"));
  EXPECT_EQ(0xC5D0, map("HANGUL SYLLABLE E"));
  EXPECT_EQ(0xC774, map("HANGUL SYLLABLE I"));

  EXPECT_EQ(0x1F984, map("UNICORN FACE"));
  EXPECT_EQ(0x00640, map("ARABIC TATWEEL"));
  EXPECT_EQ(0x02C05, map("GLAGOLITIC CAPITAL LETTER YESTU"));
  EXPECT_EQ(0x13000, map("EGYPTIAN HIEROGLYPH A001"));
  EXPECT_EQ(0x02235, map("BECAUSE"));
  EXPECT_EQ(0x1F514, map("BELL"));
  EXPECT_EQ(0x1F9A9, map("FLAMINGO"));
  EXPECT_EQ(0x1F402, map("OX")); // 2 characters
  EXPECT_EQ(0xFBF9, map("ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA "
                        "ABOVE WITH ALEF MAKSURA ISOLATED FORM"));

  // Aliases
  EXPECT_EQ(0x0000, map("NULL"));
  EXPECT_EQ(0x0007, map("ALERT"));
  EXPECT_EQ(0x0009, map("HORIZONTAL TABULATION"));
  EXPECT_EQ(0x0009, map("CHARACTER TABULATION"));
  EXPECT_EQ(0x000A, map("LINE FEED"));
  EXPECT_EQ(0x000A, map("NEW LINE"));
  EXPECT_EQ(0x0089, map("CHARACTER TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x0089, map("HORIZONTAL TABULATION WITH JUSTIFICATION"));
  EXPECT_EQ(0x2118,
            map("WEIERSTRASS ELLIPTIC FUNCTION"));     // correction
  EXPECT_EQ(0x2118, map("SCRIPT CAPITAL P"));          // correction
  EXPECT_EQ(0xFEFF, map("BYTE ORDER MARK"));           // alternate
  EXPECT_EQ(0xFEFF, map("ZERO WIDTH NO-BREAK SPACE")); // alternate
  EXPECT_EQ(0xFEFF, map("ZERO WIDTH NO BREAK SPACE")); // alternate

  // Should perform loose matching
  EXPECT_EQ(0xFFFFFFFF, map(""));
  EXPECT_EQ(0xFFFFFFFF, map("NOT A UNICODE CHARACTER"));
  EXPECT_EQ(0x0001F984, map("unicorn face"));
  EXPECT_EQ(0x0001F984, map("UNICORN FaCE"));
  EXPECT_EQ(0x0001F984, map("UNICORNFaCE"));
  EXPECT_EQ(0xFFFFFFFF, map("UNICORN"));
  EXPECT_EQ(0xC774, map("HANGUL SYLLABLE i"));
  EXPECT_EQ(0xC774, map("hANGUL SYLLABLE i"));
  EXPECT_EQ(0xC774, map("HANGULSYLLABLEI"));
  EXPECT_EQ(0xFFFFFFFF, map("HANGUL SYLLABLE"));

  EXPECT_EQ(0x2FA1D, map("cJK COMPATIBILITY IDEOGRAPH-2FA1D"));
  EXPECT_EQ(0x2FA1D, map("CJK COMPATIBILITY IDEOGRAPH-2FA1d"));
  EXPECT_EQ(0x2FA1D, map("CJK COMPATIBILITY IDEOGRAPH 2FA1D"));
  EXPECT_EQ(0x0000FEFF, map("ZERO WIDTH NO BREAK SPACE"));

  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH-NOTANUMBER"));

  EXPECT_EQ(0xFFFFFFFF, map("CJK COMPATIBILITY IDEOGRAPH-1"));
}

} // namespace
} // namespace unicode
} // namespace sys
} // namespace llvm
