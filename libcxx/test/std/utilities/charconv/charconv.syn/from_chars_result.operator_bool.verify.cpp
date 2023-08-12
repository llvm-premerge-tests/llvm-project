//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <charconv>

// struct from_chars_result
//   constexpr explicit operator bool() const noexcept { return ec == errc{}; }

#include <charconv>

void test() {
  // expected-error@+1 {{no viable conversion from 'std::from_chars_result' to 'bool'}}
  [[maybe_unused]] bool implicit = std::from_chars_result{nullptr, std::errc{}};
  if (std::from_chars_result{nullptr, std::errc{}}) {
  }
}
