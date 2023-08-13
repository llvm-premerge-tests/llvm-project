//===----------- TripleUtils.cpp - TripleUtils unit tests
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TripleUtils.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(TripleUtilsTest, CheckArchBitWidth) {
  Triple T;
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::arm);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::hexagon);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::mips);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::mips64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::msp430);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_TRUE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::ppc);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::ppc64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::x86);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::x86_64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::amdil);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::amdil64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::hsail);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::hsail64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::spir);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::spir64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::spirv32);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::spirv64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::sparc);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::sparcel);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::sparcv9);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::wasm32);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::wasm64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::avr);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_TRUE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::lanai);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::riscv32);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::riscv64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::csky);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::loongarch32);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::loongarch64);
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::dxil);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));

  T.setArch(Triple::xtensa);
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
  EXPECT_FALSE(llvm::TripleUtils::isArch16Bit(T));
}

TEST(TripleUtilsTest, CheckArchBitWidthUsingOSVersion) {
  Triple T;
  VersionTuple Version;

  T = Triple("i386-apple-darwin9");
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("x86_64-apple-darwin9");
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("x86_64-apple-macosx");
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("x86_64-apple-macosx10.7");
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("x86_64-apple-macos11.0");
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("arm64-apple-macosx11.5.8");
  EXPECT_FALSE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("armv7-apple-ios");
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("armv7-apple-ios7.0");
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));

  T = Triple("arm64_32-apple-ios");
  EXPECT_TRUE(llvm::TripleUtils::isArch32Bit(T));
}

} // namespace