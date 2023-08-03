//===-- Utilities to convert integral values to string ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts an integer to a string.
//
// By default, the string is written to an internal buffer and accessed via the
// 'string_view' operator.
//
//   const IntegerTo<DecString, int> buffer(42);
//   const cpp::string_view view(buffer);
//
// The buffer is allocated on the stack and its size is so that the conversion
// always succeeds.
//
// It is also possible to write the data to a preallocated buffer, but in this
// case the operation may fail.
//
//   char buffer[8];
//   if( auto maybe_view = IntegerTo<DecString, int>::write(42, buffer)) {
//     const cpp::string_view view = *maybe_view;
//   }
//
// The first template parameter defines how the integer is formatted.
// Available default are 'BinString', 'OctString', 'DecString' and 'HexString'.
//
// For 'BinString', 'OctString' and 'HexString' the value is always interpreted
// as a positive type but 'DecString' will honor negative values. e.g.,
//
//   IntegerTo<BinString, int8_t>(-1) // "11111111"
//   IntegerTo<OctString, int8_t>(-1) // "377"
//   IntegerTo<DecString, int8_t>(-1) // "-1"
//   IntegerTo<HexString, int8_t>(-1) // "ff"
//
// Additionnally, the default can be changed by navigating the subtypes:
//  - WithPrefix    : Adds "0b", "0", "0x" for binary, octal and hexadecimal
//  - WithWidth<XX> : Pad string to XX characters filling leading digits with 0
//  - Uppercase     : Use uppercase letters (only for HexString)
//  - WithSign      : Prepend '+' for positive values (only for DecString)
//
// Examples
// --------
//   IntegerTo<DecString::WithWidth<2>::WithSign, int8_t>(0)     // "+00"
//   IntegerTo<DecString::WithWidth<2>::WithSign, int8_t>(-1)    // "-01"
//   IntegerTo<HexString::WithPrefix::Uppercase, uint8_t>(255)   // "0xFF"
//   IntegerTo<HexString::WithWidth<8>::Uppercase, uint8_t>(255) // "000000FF"
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
#define LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H

#include <stdint.h>

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/common.h"

namespace __llvm_libc {

namespace details {

template <uint8_t base, bool prefix = false, bool is_uppercase = false,
          size_t min_digits = 1, bool force_sign = false>
struct Config {
  static constexpr size_t MIN_DIGITS = min_digits;
  static constexpr uint8_t BASE = base;
  static constexpr bool IS_UPPERCASE = is_uppercase;
  static constexpr bool PREFIX = prefix;
  static constexpr char FORCE_SIGN = force_sign;

  template <size_t value>
  using WithWidth = Config<BASE, PREFIX, IS_UPPERCASE, value, FORCE_SIGN>;
  using WithPrefix = Config<BASE, true, IS_UPPERCASE, MIN_DIGITS, FORCE_SIGN>;
  using Uppercase = Config<BASE, PREFIX, true, MIN_DIGITS, FORCE_SIGN>;
  using WithSign = Config<BASE, PREFIX, IS_UPPERCASE, MIN_DIGITS, true>;

  // Invariants
  static constexpr uint8_t NUMERICAL_DIGITS = 10;
  static constexpr uint8_t ALPHA_DIGITS = 26;
  static constexpr uint8_t MAX_DIGIT = NUMERICAL_DIGITS + ALPHA_DIGITS;
  static_assert(BASE > 1 && BASE <= MAX_DIGIT);
};

} // namespace details

using BinString = details::Config<2>;
using OctString = details::Config<8>;
using DecString = details::Config<10>;
using HexString = details::Config<16>;

template <typename Conf, typename T> class IntegerTo {
  // We size the string buffer for base 10 using an approximation algorithm:
  //
  //   size = ceil(sizeof(T) * 5 / 2)
  //
  // If sizeof(T) is 1, then size is 3 (actually need 3)
  // If sizeof(T) is 2, then size is 5 (actually need 5)
  // If sizeof(T) is 4, then size is 10 (actually need 10)
  // If sizeof(T) is 8, then size is 20 (actually need 20)
  // If sizeof(T) is 16, then size is 40 (actually need 39)
  //
  // NOTE: The ceil operation is actually implemented as
  //     floor(((sizeof(T) * 5) + 1)/2)
  // where floor operation is just integer division.
  //
  // This estimation grows slightly faster than the actual value, but the
  // overhead is small enough to tolerate. In the actual formula below, we
  // add an additional byte to accommodate the '-' sign in case of signed
  // integers.
  // For other bases, we approximate by rounding down to the nearest power of
  // two base, since the space needed is easy to calculate and it won't
  // overestimate by too much.
  LIBC_INLINE static constexpr size_t compute_digit_size() {
    if constexpr (Conf::BASE == 10)
      return (sizeof(T) * 5 + 1) / 2;
    auto floor_log_2 = [](size_t num) -> size_t {
      size_t i = 0;
      for (; num > 1; num /= 2)
        ++i;
      return i;
    };
    constexpr size_t BITS_PER_DIGIT = floor_log_2(Conf::BASE);
    return ((sizeof(T) * 8 + (BITS_PER_DIGIT - 1)) / BITS_PER_DIGIT);
  }
  LIBC_INLINE static constexpr size_t max(size_t a, size_t b) {
    return a < b ? b : a;
  }
  LIBC_INLINE static char digit_char(uint8_t digit) {
    if (digit < 10)
      return '0' + digit;
    return (Conf::IS_UPPERCASE ? 'A' : 'a') + (digit - 10);
  }
  // constants
  using UNSIGNED_T = cpp::make_unsigned_t<T>;
  static constexpr size_t BUFFER_SIZE =
      max(compute_digit_size(), Conf::MIN_DIGITS) + (Conf::BASE == 10 ? 1 : 0) +
      (Conf::PREFIX ? 2 : 0);
  // Invariants
  static_assert(cpp::is_integral_v<T>);
  static_assert(BUFFER_SIZE > 1);

  // Internal abstraction to handle different kind of buffers.
  struct Sink {
    char *const begin;
    char *const end;
    char *ptr;

    explicit Sink(cpp::span<char> buffer)
        : begin(buffer.begin()), end(buffer.end()), ptr(end) {}

    char last_char() const { return *ptr; }
    size_t size() const { return end - ptr; }
    size_t empty() const { return size() == 0; }

    void push_front(char c) {
      if (buffer_overrun())
        return;
      --ptr;
      *ptr = c;
    }

    // Returns whether the sink is still in a valid state.
    operator bool() const { return !buffer_overrun(); }

  private:
    bool buffer_overrun() const { return ptr < begin; }
  };

  // Internal buffer.
  struct Buffer {
    cpp::array<char, BUFFER_SIZE> array;
    size_t size = 0;
    cpp::span<char> as_span() { return cpp::span<char>(array); }
  };

  LIBC_INLINE static void convert(UNSIGNED_T value, Sink &sink) {
    for (; sink && value != 0; value /= Conf::BASE) {
      const uint8_t digit(value % Conf::BASE);
      sink.push_front(digit_char(digit));
    }
  }

  LIBC_INLINE static void process(T value, Sink &sink) {
    bool is_negative = false;
    if constexpr (Conf::BASE == 10) {
      is_negative = value < 0;
      convert(is_negative ? -value : value, sink);
    } else {
      convert(cpp::bit_cast<UNSIGNED_T>(value), sink);
    }
    // width
    while (sink && sink.size() < Conf::MIN_DIGITS)
      sink.push_front('0');
    // sign
    if constexpr (Conf::BASE == 10) {
      if (is_negative)
        sink.push_front('-');
      if (!is_negative && Conf::FORCE_SIGN)
        sink.push_front('+');
    }
    // prefix
    if (Conf::PREFIX) {
      if (Conf::BASE == 2) {
        sink.push_front('b');
        sink.push_front('0');
      }
      if (Conf::BASE == 16) {
        sink.push_front('x');
        sink.push_front('0');
      }
      if (Conf::BASE == 8 && (sink.empty() || sink.last_char() != '0')) {
        sink.push_front('0');
      }
    }
  }

  Buffer buffer;

public:
  explicit IntegerTo(T value) {
    Sink sink(buffer.as_span());
    process(value, sink);
    buffer.size = sink.size();
  }

  size_t size() const { return buffer.size; }

  // Prevent returning dangling string_view.
  explicit operator cpp::string_view() && = delete;
  explicit operator cpp::string_view() const & {
    return {buffer.array.end() - size(), size()};
  }

  LIBC_INLINE static constexpr size_t buffer_size() { return BUFFER_SIZE; }

  LIBC_INLINE static cpp::optional<cpp::string_view>
  write_to(cpp::span<char> span, T value) {
    Sink sink(span);
    process(value, sink);
    if (sink)
      return cpp::string_view(span.end() - sink.size(), sink.size());
    return cpp::nullopt;
  }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_INTEGER_TO_STRING_H
