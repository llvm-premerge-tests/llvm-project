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

// constexpr iterator(iterator<!Const> i)
//   requires Const && convertible_to<iterator_t<V>, iterator_t<Base>>;

#include <cassert>
#include <ranges>

#include "test_iterators.h"

constexpr bool test() { return true; }

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
