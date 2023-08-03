//===-- Unittests for IntegerToString -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/UInt.h"
#include "src/__support/UInt128.h"
#include "src/__support/integer_to_string.h"

#include "test/UnitTest/Test.h"

#include "limits.h"

using __llvm_libc::BinString;
using __llvm_libc::DecString;
using __llvm_libc::HexString;
using __llvm_libc::IntegerTo;
using __llvm_libc::OctString;
using __llvm_libc::cpp::span;
using __llvm_libc::cpp::string_view;

#define EXPECT(type, value, string_value)                                      \
  {                                                                            \
    const type buffer(value);                                                  \
    EXPECT_EQ(static_cast<string_view>(buffer), string_view(string_value));    \
  }

TEST(LlvmLibcIntegerToStringTest, UINT8) {
  using type = IntegerTo<DecString, uint8_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, UINT8_MAX, "255");
  EXPECT(type, -1, "255");
}

TEST(LlvmLibcIntegerToStringTest, INT8) {
  using type = IntegerTo<DecString, int8_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, -12, "-12");
  EXPECT(type, -123, "-123");
  EXPECT(type, INT8_MAX, "127");
  EXPECT(type, INT8_MIN, "-128");
}

TEST(LlvmLibcIntegerToStringTest, UINT16) {
  using type = IntegerTo<DecString, uint16_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, UINT16_MAX, "65535");
  EXPECT(type, -1, "65535");
}

TEST(LlvmLibcIntegerToStringTest, INT16) {
  using type = IntegerTo<DecString, int16_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, -1, "-1");
  EXPECT(type, -12, "-12");
  EXPECT(type, -123, "-123");
  EXPECT(type, -1234, "-1234");
  EXPECT(type, -12345, "-12345");
  EXPECT(type, INT16_MAX, "32767");
  EXPECT(type, INT16_MIN, "-32768");
}

TEST(LlvmLibcIntegerToStringTest, UINT32) {
  using type = IntegerTo<DecString, uint32_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, 123456, "123456");
  EXPECT(type, 1234567, "1234567");
  EXPECT(type, 12345678, "12345678");
  EXPECT(type, 123456789, "123456789");
  EXPECT(type, 1234567890, "1234567890");
  EXPECT(type, UINT32_MAX, "4294967295");
  EXPECT(type, -1, "4294967295");
}

TEST(LlvmLibcIntegerToStringTest, INT32) {
  using type = IntegerTo<DecString, int32_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, 123456, "123456");
  EXPECT(type, 1234567, "1234567");
  EXPECT(type, 12345678, "12345678");
  EXPECT(type, 123456789, "123456789");
  EXPECT(type, 1234567890, "1234567890");
  EXPECT(type, -1, "-1");
  EXPECT(type, -12, "-12");
  EXPECT(type, -123, "-123");
  EXPECT(type, -1234, "-1234");
  EXPECT(type, -12345, "-12345");
  EXPECT(type, -123456, "-123456");
  EXPECT(type, -1234567, "-1234567");
  EXPECT(type, -12345678, "-12345678");
  EXPECT(type, -123456789, "-123456789");
  EXPECT(type, -1234567890, "-1234567890");
  EXPECT(type, INT32_MAX, "2147483647");
  EXPECT(type, INT32_MIN, "-2147483648");
}

TEST(LlvmLibcIntegerToStringTest, UINT64) {
  using type = IntegerTo<DecString, uint64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, 123456, "123456");
  EXPECT(type, 1234567, "1234567");
  EXPECT(type, 12345678, "12345678");
  EXPECT(type, 123456789, "123456789");
  EXPECT(type, 1234567890, "1234567890");
  EXPECT(type, 1234567890123456789, "1234567890123456789");
  EXPECT(type, UINT64_MAX, "18446744073709551615");
  EXPECT(type, -1, "18446744073709551615");
}

TEST(LlvmLibcIntegerToStringTest, INT64) {
  using type = IntegerTo<DecString, int64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 1, "1");
  EXPECT(type, 12, "12");
  EXPECT(type, 123, "123");
  EXPECT(type, 1234, "1234");
  EXPECT(type, 12345, "12345");
  EXPECT(type, 123456, "123456");
  EXPECT(type, 1234567, "1234567");
  EXPECT(type, 12345678, "12345678");
  EXPECT(type, 123456789, "123456789");
  EXPECT(type, 1234567890, "1234567890");
  EXPECT(type, 1234567890123456789, "1234567890123456789");
  EXPECT(type, -1, "-1");
  EXPECT(type, -12, "-12");
  EXPECT(type, -123, "-123");
  EXPECT(type, -1234, "-1234");
  EXPECT(type, -12345, "-12345");
  EXPECT(type, -123456, "-123456");
  EXPECT(type, -1234567, "-1234567");
  EXPECT(type, -12345678, "-12345678");
  EXPECT(type, -123456789, "-123456789");
  EXPECT(type, -1234567890, "-1234567890");
  EXPECT(type, -1234567890123456789, "-1234567890123456789");
  EXPECT(type, INT64_MAX, "9223372036854775807");
  EXPECT(type, INT64_MIN, "-9223372036854775808");
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_8) {
  using type = IntegerTo<OctString, int64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 012345, "12345");
  EXPECT(type, 0123456701234567012345, "123456701234567012345");
  EXPECT(type, 01777777777777777777777, "1777777777777777777777");
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_16) {
  using type = IntegerTo<HexString, uint64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 0x12345, "12345");
  EXPECT(type, 0x123456789abcdef, "123456789abcdef");
  EXPECT(type, 0xffffffffffffffff, "ffffffffffffffff");
  using TYPE = IntegerTo<HexString::Uppercase, uint64_t>;
  EXPECT(TYPE, 0x123456789abcdef, "123456789ABCDEF");
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_2) {
  using type = IntegerTo<BinString, uint64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 0b111100001100, "111100001100");
  EXPECT(type, 0b100100011101010111100, "100100011101010111100");
  EXPECT(type, 0xffffffffffffffff,
         "1111111111111111111111111111111111111111111111111111111111111111");
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_36) {
  using type = IntegerTo<__llvm_libc::details::Config<36>, uint64_t>;
  EXPECT(type, 0, "0");
  EXPECT(type, 12345, "9ix");
  EXPECT(type, 1047601316295595, "abcdefghij");
  EXPECT(type, 2092218013456445, "klmnopqrst");
  EXPECT(type, 0xffffffffffffffff, "3w5e11264sgsf");

  using TYPE = IntegerTo<__llvm_libc::details::Config<36>::Uppercase, uint64_t>;
  EXPECT(TYPE, 1867590395, "UVWXYZ");
}

TEST(LlvmLibcIntegerToStringTest, UINT128_Base_16) {
  using type = IntegerTo<HexString::WithWidth<32>, UInt128>;
  EXPECT(type, 0, "00000000000000000000000000000000");
  EXPECT(type, 0x12345, "00000000000000000000000000012345");
  EXPECT(type, static_cast<UInt128>(0x1234) << 112,
         "12340000000000000000000000000000");
  EXPECT(type, static_cast<UInt128>(0x1234) << 48,
         "00000000000000001234000000000000");
  EXPECT(type, static_cast<UInt128>(0x1234) << 52,
         "00000000000000012340000000000000");
}

TEST(LlvmLibcIntegerToStringTest, UINT256_Base_16) {
  using UInt256 = __llvm_libc::cpp::UInt<256>;
  using type = IntegerTo<HexString::WithWidth<64>, UInt256>;
  EXPECT(type, static_cast<UInt256>(0),
         "0000000000000000000000000000000000000000000000000000000000000000");
  EXPECT(type, static_cast<UInt256>(0x12345),
         "0000000000000000000000000000000000000000000000000000000000012345");
  EXPECT(type, static_cast<UInt256>(0x1234) << 112,
         "0000000000000000000000000000000012340000000000000000000000000000");
  EXPECT(type, static_cast<UInt256>(0x1234) << 116,
         "0000000000000000000000000000000123400000000000000000000000000000");
  EXPECT(type, static_cast<UInt256>(0x1234) << 240,
         "1234000000000000000000000000000000000000000000000000000000000000");
}

TEST(LlvmLibcIntegerToStringTest, NegativeInterpretedAsPositive) {
  using BIN = IntegerTo<BinString, int8_t>;
  using OCT = IntegerTo<OctString, int8_t>;
  using DEC = IntegerTo<DecString, int8_t>;
  using HEX = IntegerTo<HexString, int8_t>;
  EXPECT(BIN, -1, "11111111");
  EXPECT(OCT, -1, "377");
  EXPECT(DEC, -1, "-1"); // Only DEC format negatve values
  EXPECT(HEX, -1, "ff");
}

TEST(LlvmLibcIntegerToStringTest, Width) {
  using BIN = IntegerTo<BinString::WithWidth<4>, uint8_t>;
  using OCT = IntegerTo<OctString::WithWidth<4>, uint8_t>;
  using DEC = IntegerTo<DecString::WithWidth<4>, uint8_t>;
  using HEX = IntegerTo<HexString::WithWidth<4>, uint8_t>;
  EXPECT(BIN, 1, "0001");
  EXPECT(HEX, 1, "0001");
  EXPECT(OCT, 1, "0001");
  EXPECT(DEC, 1, "0001");
}

TEST(LlvmLibcIntegerToStringTest, Prefix) {
  using BIN = IntegerTo<BinString::WithPrefix, uint8_t>;
  using OCT = IntegerTo<OctString::WithPrefix, uint8_t>;
  using DEC = IntegerTo<DecString::WithPrefix, uint8_t>;
  using HEX = IntegerTo<HexString::WithPrefix, uint8_t>;
  EXPECT(BIN, 1, "0b1");
  EXPECT(HEX, 1, "0x1");
  EXPECT(OCT, 1, "01");
  EXPECT(DEC, 1, "1");
  EXPECT(OCT, 0, "0"); // Zero is not prefixed for octal
}

TEST(LlvmLibcIntegerToStringTest, Uppercase) {
  using HEX = IntegerTo<HexString::Uppercase, uint64_t>;
  EXPECT(HEX, 0xDEADC0DE, "DEADC0DE");
}

TEST(LlvmLibcIntegerToStringTest, Sign) {
  using BIN = IntegerTo<BinString::WithSign, uint8_t>;
  using OCT = IntegerTo<OctString::WithSign, uint8_t>;
  using DEC = IntegerTo<DecString::WithSign, uint8_t>;
  using HEX = IntegerTo<HexString::WithSign, uint8_t>;
  EXPECT(BIN, 1, "1");
  EXPECT(HEX, 1, "1");
  EXPECT(OCT, 1, "1");
  EXPECT(DEC, 1, "+1"); // Force sign only work with DEC
}

TEST(LlvmLibcIntegerToStringTest, BufferOverrun) {
  { // Writing '0' in an empty buffer requiring zero digits works
    const auto view =
        IntegerTo<DecString::WithWidth<0>, int>::write_to(span<char>(), 0);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(*view, string_view(""));
  }
  char buffer[1];
  { // Writing '1' in a buffer of one char works
    const auto view = IntegerTo<DecString, int>::write_to(buffer, 1);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(*view, string_view("1"));
  }
  { // Writing '11' in a buffer of one char fails
    const auto view = IntegerTo<DecString, int>::write_to(buffer, 11);
    ASSERT_FALSE(view.has_value());
  }
}
