//===--OrganizeImportsTest.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "TweakTesting.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(OrganizeImports);

TEST_F(OrganizeImportsTest, Prepare) {
  struct {
    llvm::StringRef Code;
    bool IsAvailable;
    const std::vector<std::string> RequestedActionKinds;
  } Cases[] = {{
                   R"cpp(
#include "Te^stTU.h"
)cpp",
                   true,
                   {CodeAction::SOURCE_KIND.str()}},
               {"void foo(^) {}", true, {CodeAction::SOURCE_KIND.str()}},
               {
                   R"cpp(
#include "Te^stTU.h"
)cpp",
                   true,
                   {}},
               {"void foo(^) {}", false, {}}};

  for (auto Case : Cases) {
    llvm::Annotations A{Case.Code};
    auto AST = build(A.code());
    for (const auto &P : A.points())
      EXPECT_EQ(Case.IsAvailable,
                isAvailable(AST, {P, P}, Case.RequestedActionKinds))
          << decorate(A.code(), P);
  }
}

TEST_F(OrganizeImportsTest, Apply) {
  Header = "void foo();";
  struct {
    llvm::StringRef Code;
    llvm::StringRef Header;
    llvm::StringRef Expected;
  } Cases[] = {{// Remove unused include.
                R"cpp(
#include "TestTU.h"
void foo() {}
void b^ar() {
    foo();
}
)cpp",
                R"cpp(
#pragma once
)cpp",
                R"cpp(
void foo() {}
void bar() {
    foo();
}
)cpp"},
               {// Add missing include.
                R"cpp(
void b^ar() {
    foo();
}
)cpp",
                R"cpp(
#pragma once
void foo();
)cpp",
                R"cpp(
#include "TestTU.h"
void bar() {
    foo();
}
)cpp"},
               {// Replace unused include with missing.
                R"cpp(
#include "foo.h"
void b^ar() {
    foo();
}
)cpp",
                R"cpp(
#pragma once
void foo();
)cpp",
                R"cpp(
#include "TestTU.h"
void bar() {
    foo();
}
)cpp"}};
  for (auto C : Cases) {
    Header = C.Header;
    ExtraFiles["foo.h"] = "#pragma once";
    EXPECT_EQ(C.Expected,
              apply(C.Code, nullptr, {CodeAction::SOURCE_KIND.str()}))
        << C.Expected;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
