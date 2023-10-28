//===- unittest/Format/ObjCPropertyAttributeOrderFixerTest.cpp - unit tests
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../lib/Format/ObjCPropertyAttributeOrderFixer.h"
#include "FormatTestBase.h"
#include "TestLexer.h"

#define DEBUG_TYPE "format-objc-property-attribute-order-fixer-test"

namespace clang {
namespace format {
namespace test {
namespace {

#define CHECK_PARSE(TEXT, FIELD, VALUE)                                        \
  EXPECT_NE(VALUE, Style.FIELD) << "Initial value already the same!";          \
  EXPECT_EQ(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

#define FAIL_PARSE(TEXT, FIELD, VALUE)                                         \
  EXPECT_NE(0, parseConfiguration(TEXT, &Style).value());                      \
  EXPECT_EQ(VALUE, Style.FIELD) << "Unexpected value after parsing!"

class ObjCPropertyAttributeOrderFixerTest : public FormatTestBase {
protected:
  TokenList annotate(llvm::StringRef Code,
                     const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).annotate(Code);
  }

  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
};

TEST_F(ObjCPropertyAttributeOrderFixerTest, ParsesStyleOption) {
  FormatStyle Style = {};
  Style.Language = FormatStyle::LK_ObjC;

  CHECK_PARSE("ObjCPropertyAttributeOrder: [class]", ObjCPropertyAttributeOrder,
              std::vector<std::string>({"class"}));

  CHECK_PARSE("ObjCPropertyAttributeOrder: ["
              "class, direct, atomic, nonatomic, "
              "assign, retain, strong, copy, weak, unsafe_unretained, "
              "readonly, readwrite, getter, setter, "
              "nullable, nonnull, null_resettable, null_unspecified"
              "]",
              ObjCPropertyAttributeOrder,
              std::vector<std::string>({
                  "class",
                  "direct",
                  "atomic",
                  "nonatomic",
                  "assign",
                  "retain",
                  "strong",
                  "copy",
                  "weak",
                  "unsafe_unretained",
                  "readonly",
                  "readwrite",
                  "getter",
                  "setter",
                  "nullable",
                  "nonnull",
                  "null_resettable",
                  "null_unspecified",
              }));
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, SortsSpecifiedAttributes) {
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"a", "b", "c"};

  verifyFormat("@property() int p;", Style);

  // One: shouldn't move.
  verifyFormat("@property(a) int p;", Style);
  verifyFormat("@property(b) int p;", Style);
  verifyFormat("@property(c) int p;", Style);

  // Two in correct order already: no change.
  verifyFormat("@property(a, b) int p;", Style);
  verifyFormat("@property(a, c) int p;", Style);
  verifyFormat("@property(b, c) int p;", Style);

  // Three in correct order already: no change.
  verifyFormat("@property(a, b, c) int p;", Style);

  // Two wrong order.
  verifyFormat("@property(a, b) int p;", "@property(b, a) int p;", Style);
  verifyFormat("@property(a, c) int p;", "@property(c, a) int p;", Style);
  verifyFormat("@property(b, c) int p;", "@property(c, b) int p;", Style);

  // Three wrong order.
  verifyFormat("@property(a, b, c) int p;", "@property(b, a, c) int p;", Style);
  verifyFormat("@property(a, b, c) int p;", "@property(c, b, a) int p;", Style);
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, SortsAttributesWithValues) {
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"a", "getter", "c"};

  // No change
  verifyFormat("@property(getter=G, c) int p;", Style);
  verifyFormat("@property(a, getter=G) int p;", Style);
  verifyFormat("@property(a, getter=G, c) int p;", Style);

  // Reorder
  verifyFormat("@property(getter=G, c) int p;", "@property(c, getter=G) int p;",
               Style);
  verifyFormat("@property(a, getter=G) int p;", "@property(getter=G, a) int p;",
               Style);
  verifyFormat("@property(a, getter=G, c) int p;",
               "@property(getter=G, c, a) int p;", Style);

  // Multiple set properties, including ones not recognized
  verifyFormat("@property(a=A, c=C, x=X, y=Y) int p;",
               "@property(c=C, x=X, y=Y, a=A) int p;", Style);
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, SortsUnspecifiedAttributesToBack) {
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"a", "b", "c"};

  verifyFormat("@property(x) int p;", Style);

  // No change in order.
  verifyFormat("@property(a, x, y) int p;", Style);
  verifyFormat("@property(b, x, y) int p;", Style);
  verifyFormat("@property(a, b, c, x, y) int p;", Style);

  // Reorder one unrecognized one.
  verifyFormat("@property(a, x) int p;", "@property(x, a) int p;", Style);

  // Prove the unrecognized ones have a stable sort order
  verifyFormat("@property(a, b, x, y) int p;", "@property(x, b, y, a) int p;",
               Style);
  verifyFormat("@property(a, b, y, x) int p;", "@property(y, b, x, a) int p;",
               Style);
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, RemovesDuplicateAttributes) {
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"a", "b", "c"};

  verifyFormat("@property(a) int p;", "@property(a, a) int p;", Style);
  verifyFormat("@property(a) int p;", "@property(a, a, a, a) int p;", Style);

  verifyFormat("@property(a, b, c) int p;",
               "@property(c, b, a, b, a, c) int p;", Style);

  verifyFormat("@property(a, b, c, x, y) int p;",
               "@property(c, x, b, a, y, b, a, c, y) int p;", Style);
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, HandlesAllAttributes) {
  // 'class' is the only attribute that is a keyword, so make sure it works too.
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"FIRST",
                                      "class",
                                      "direct",
                                      "atomic",
                                      "nonatomic",
                                      "assign",
                                      "retain",
                                      "strong",
                                      "copy",
                                      "weak",
                                      "unsafe_unretained",
                                      "readonly",
                                      "readwrite",
                                      "getter",
                                      "setter",
                                      "nullable",
                                      "nonnull",
                                      "null_resettable",
                                      "null_unspecified",
                                      "LAST"};

  // No change: specify all attributes in the correct order.
  verifyFormat("@property(class, LAST) int p;", Style);
  verifyFormat("@property(direct, LAST) int p;", Style);
  verifyFormat("@property(atomic, LAST) int p;", Style);
  verifyFormat("@property(nonatomic, LAST) int p;", Style);
  verifyFormat("@property(assign, LAST) int p;", Style);
  verifyFormat("@property(retain, LAST) int p;", Style);
  verifyFormat("@property(strong, LAST) int p;", Style);
  verifyFormat("@property(copy, LAST) int p;", Style);
  verifyFormat("@property(weak, LAST) int p;", Style);
  verifyFormat("@property(unsafe_unretained, LAST) int p;", Style);
  verifyFormat("@property(readonly, LAST) int p;", Style);
  verifyFormat("@property(readwrite, LAST) int p;", Style);
  verifyFormat("@property(getter, LAST) int p;", Style);
  verifyFormat("@property(setter, LAST) int p;", Style);
  verifyFormat("@property(nullable, LAST) int p;", Style);
  verifyFormat("@property(nonnull, LAST) int p;", Style);
  verifyFormat("@property(null_resettable, LAST) int p;", Style);
  verifyFormat("@property(null_unspecified, LAST) int p;", Style);

  verifyFormat("@property(FIRST, class) int p;", Style);
  verifyFormat("@property(FIRST, direct) int p;", Style);
  verifyFormat("@property(FIRST, atomic) int p;", Style);
  verifyFormat("@property(FIRST, nonatomic) int p;", Style);
  verifyFormat("@property(FIRST, assign) int p;", Style);
  verifyFormat("@property(FIRST, retain) int p;", Style);
  verifyFormat("@property(FIRST, strong) int p;", Style);
  verifyFormat("@property(FIRST, copy) int p;", Style);
  verifyFormat("@property(FIRST, weak) int p;", Style);
  verifyFormat("@property(FIRST, unsafe_unretained) int p;", Style);
  verifyFormat("@property(FIRST, readonly) int p;", Style);
  verifyFormat("@property(FIRST, readwrite) int p;", Style);
  verifyFormat("@property(FIRST, getter) int p;", Style);
  verifyFormat("@property(FIRST, setter) int p;", Style);
  verifyFormat("@property(FIRST, nullable) int p;", Style);
  verifyFormat("@property(FIRST, nonnull) int p;", Style);
  verifyFormat("@property(FIRST, null_resettable) int p;", Style);
  verifyFormat("@property(FIRST, null_unspecified) int p;", Style);

  verifyFormat("@property(FIRST, class, LAST) int p;", Style);
  verifyFormat("@property(FIRST, direct, LAST) int p;", Style);
  verifyFormat("@property(FIRST, atomic, LAST) int p;", Style);
  verifyFormat("@property(FIRST, nonatomic, LAST) int p;", Style);
  verifyFormat("@property(FIRST, assign, LAST) int p;", Style);
  verifyFormat("@property(FIRST, retain, LAST) int p;", Style);
  verifyFormat("@property(FIRST, strong, LAST) int p;", Style);
  verifyFormat("@property(FIRST, copy, LAST) int p;", Style);
  verifyFormat("@property(FIRST, weak, LAST) int p;", Style);
  verifyFormat("@property(FIRST, unsafe_unretained, LAST) int p;", Style);
  verifyFormat("@property(FIRST, readonly, LAST) int p;", Style);
  verifyFormat("@property(FIRST, readwrite, LAST) int p;", Style);
  verifyFormat("@property(FIRST, getter, LAST) int p;", Style);
  verifyFormat("@property(FIRST, setter, LAST) int p;", Style);
  verifyFormat("@property(FIRST, nullable, LAST) int p;", Style);
  verifyFormat("@property(FIRST, nonnull, LAST) int p;", Style);
  verifyFormat("@property(FIRST, null_resettable, LAST) int p;", Style);
  verifyFormat("@property(FIRST, null_unspecified, LAST) int p;", Style);

  // Reorder: put 'FIRST' and/or 'LAST' in the wrong spot.
  verifyFormat("@property(class, LAST) int p;", "@property(LAST, class) int p;",
               Style);
  verifyFormat("@property(direct, LAST) int p;",
               "@property(LAST, direct) int p;", Style);
  verifyFormat("@property(atomic, LAST) int p;",
               "@property(LAST, atomic) int p;", Style);
  verifyFormat("@property(nonatomic, LAST) int p;",
               "@property(LAST, nonatomic) int p;", Style);
  verifyFormat("@property(assign, LAST) int p;",
               "@property(LAST, assign) int p;", Style);
  verifyFormat("@property(retain, LAST) int p;",
               "@property(LAST, retain) int p;", Style);
  verifyFormat("@property(strong, LAST) int p;",
               "@property(LAST, strong) int p;", Style);
  verifyFormat("@property(copy, LAST) int p;", "@property(LAST, copy) int p;",
               Style);
  verifyFormat("@property(weak, LAST) int p;", "@property(LAST, weak) int p;",
               Style);
  verifyFormat("@property(unsafe_unretained, LAST) int p;",
               "@property(LAST, unsafe_unretained) int p;", Style);
  verifyFormat("@property(readonly, LAST) int p;",
               "@property(LAST, readonly) int p;", Style);
  verifyFormat("@property(readwrite, LAST) int p;",
               "@property(LAST, readwrite) int p;", Style);
  verifyFormat("@property(getter, LAST) int p;",
               "@property(LAST, getter) int p;", Style);
  verifyFormat("@property(setter, LAST) int p;",
               "@property(LAST, setter) int p;", Style);
  verifyFormat("@property(nullable, LAST) int p;",
               "@property(LAST, nullable) int p;", Style);
  verifyFormat("@property(nonnull, LAST) int p;",
               "@property(LAST, nonnull) int p;", Style);
  verifyFormat("@property(null_resettable, LAST) int p;",
               "@property(LAST, null_resettable) int p;", Style);
  verifyFormat("@property(null_unspecified, LAST) int p;",
               "@property(LAST, null_unspecified) int p;", Style);

  verifyFormat("@property(FIRST, class) int p;",
               "@property(class, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, direct) int p;",
               "@property(direct, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, atomic) int p;",
               "@property(atomic, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nonatomic) int p;",
               "@property(nonatomic, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, assign) int p;",
               "@property(assign, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, retain) int p;",
               "@property(retain, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, strong) int p;",
               "@property(strong, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, copy) int p;", "@property(copy, FIRST) int p;",
               Style);
  verifyFormat("@property(FIRST, weak) int p;", "@property(weak, FIRST) int p;",
               Style);
  verifyFormat("@property(FIRST, unsafe_unretained) int p;",
               "@property(unsafe_unretained, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, readonly) int p;",
               "@property(readonly, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, readwrite) int p;",
               "@property(readwrite, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, getter) int p;",
               "@property(getter, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, setter) int p;",
               "@property(setter, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nullable) int p;",
               "@property(nullable, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nonnull) int p;",
               "@property(nonnull, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, null_resettable) int p;",
               "@property(null_resettable, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, null_unspecified) int p;",
               "@property(null_unspecified, FIRST) int p;", Style);

  verifyFormat("@property(FIRST, class, LAST) int p;",
               "@property(LAST, class, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, direct, LAST) int p;",
               "@property(LAST, direct, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, atomic, LAST) int p;",
               "@property(LAST, atomic, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nonatomic, LAST) int p;",
               "@property(LAST, nonatomic, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, assign, LAST) int p;",
               "@property(LAST, assign, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, retain, LAST) int p;",
               "@property(LAST, retain, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, strong, LAST) int p;",
               "@property(LAST, strong, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, copy, LAST) int p;",
               "@property(LAST, copy, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, weak, LAST) int p;",
               "@property(LAST, weak, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, unsafe_unretained, LAST) int p;",
               "@property(LAST, unsafe_unretained, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, readonly, LAST) int p;",
               "@property(LAST, readonly, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, readwrite, LAST) int p;",
               "@property(LAST, readwrite, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, getter, LAST) int p;",
               "@property(LAST, getter, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, setter, LAST) int p;",
               "@property(LAST, setter, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nullable, LAST) int p;",
               "@property(LAST, nullable, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, nonnull, LAST) int p;",
               "@property(LAST, nonnull, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, null_resettable, LAST) int p;",
               "@property(LAST, null_resettable, FIRST) int p;", Style);
  verifyFormat("@property(FIRST, null_unspecified, LAST) int p;",
               "@property(LAST, null_unspecified, FIRST) int p;", Style);
}

TEST_F(ObjCPropertyAttributeOrderFixerTest, HandlesCommentsAroundAttributes) {
  FormatStyle Style = getLLVMStyle();
  Style.ObjCPropertyAttributeOrder = {"a", "b", "c"};

  // Handle zero attributes but comments.
  verifyFormat("@property(/* 1 */) int p;", Style);
  verifyFormat("@property(/* 1 */ /* 2 */) int p;", Style);

  // Handle one attribute with comments before or after.
  verifyFormat("@property(/* 1 */ a) int p;", Style);
  verifyFormat("@property(a /* 2 */) int p;", Style);
  verifyFormat("@property(/* 1 */ a /* 2 */) int p;", Style);

  // Handle reordering with comments, before or after or both.
  verifyFormat("@property(/* 1 */ a, b, x, y) int p;",
               "@property(/* 1 */ x, b, a, y) int p;", Style);

  verifyFormat("@property(a, b, x, y /* 2 */) int p;",
               "@property(x, b, a, y /* 2 */) int p;", Style);

  verifyFormat("@property(/* 1 */ a, b, x, y /* 2 */) int p;",
               "@property(/* 1 */ x, b, a, y /* 2 */) int p;", Style);

  verifyFormat("@property(/* 1 */ /* 2 */ a, b, x, y /* 3 */ /* 4 */) int p;",
               "@property(/* 1 *//* 2 */ x,b,a,y /* 3 *//* 4 */) int p;",
               Style);

  // Comments between properties cause the pass to bail.
  verifyFormat("@property(a, /* 1 */ b) int p;", Style);
  verifyFormat("@property(b, /* 1 */ a) int p;", Style);
  verifyFormat("@property(b /* 1 */, a) int p;", Style);
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
