//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

//  constexpr sentinel(sentinel<!Const> other)
//       requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <cassert>
#include <ranges>
#include <array>

#include "test_iterators.h"
#include "../types.h"

struct Sentinel {
  int i;
  constexpr Sentinel() = default;
  constexpr Sentinel(int ii) : i(ii) {}
  friend constexpr bool operator==(int*, const Sentinel&) { return true; }
};

struct ConstSentinel {
  int i;
  constexpr ConstSentinel() = default;
  constexpr ConstSentinel(int ii) : i(ii) {}
  constexpr ConstSentinel(const Sentinel& s) : i(s.i) {}
  friend constexpr bool operator==(int*, const ConstSentinel&) { return true; }
};

struct ConstSentinelRange : std::ranges::view_base {
  int i;
  constexpr ConstSentinelRange() = default;
  constexpr ConstSentinelRange(int ii) : i(ii) {}
  int* begin() const;
  Sentinel end();
  ConstSentinel end() const;
};

struct NonConvertConstSentinel {
  int i;
  constexpr NonConvertConstSentinel() = default;
  constexpr NonConvertConstSentinel(int ii) : i(ii) {}
  friend constexpr bool operator==(int*, const NonConvertConstSentinel&) { return true; }
};

struct NonConvertConstSentinelRange : std::ranges::view_base {
  int* begin() const;
  Sentinel end();
  NonConvertConstSentinel end() const;
};

// Test Constraint
static_assert(std::is_constructible_v<std::ranges::sentinel_t<const std::ranges::enumerate_view<ConstSentinelRange>>,
                                      std::ranges::sentinel_t<std::ranges::enumerate_view<ConstSentinelRange>>>);

// !Const
static_assert(!std::is_constructible_v<std::ranges::sentinel_t<std::ranges::enumerate_view<ConstSentinelRange>>,
                                       std::ranges::sentinel_t<const std::ranges::enumerate_view<ConstSentinelRange>>>);

// !convertible_to<sentinel_t<V>, sentinel_t<Base>>
static_assert(
    !std::is_constructible_v< std::ranges::sentinel_t<const std::ranges::enumerate_view<NonConvertConstSentinelRange>>,
                              std::ranges::sentinel_t<std::ranges::enumerate_view<NonConvertConstSentinelRange>>>);

// template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using EnumerateView          = std::ranges::enumerate_view<ConstSentinelRange>;
  using EnumerateSentinel      = std::ranges::sentinel_t<EnumerateView>;
  using ConstEnumerateSentinel = std::ranges::sentinel_t<const EnumerateView>;
  static_assert(!std::same_as<EnumerateSentinel, ConstEnumerateSentinel>);

  //   EnumerateSentinel s1{EnumerateView{84}};
  //   ConstEnumerateSentinel s2 = s1;

  //   assert(s2.base().i == 84);
}

constexpr bool tests() {
  test();
  // test<cpp17_input_iterator<int*>>();
  // test<cpp20_input_iterator<int*>>();
  // test<forward_iterator<int*>>();
  // test<bidirectional_iterator<int*>>();
  // test<random_access_iterator<int*>>();
  // test<contiguous_iterator<int*>>();
  // test<int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
