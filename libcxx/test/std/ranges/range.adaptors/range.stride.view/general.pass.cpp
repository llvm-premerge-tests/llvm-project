//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::stride_view

#include <ranges>

#include <cassert>

constexpr bool static_smoke_test() { return true; }

bool smoke_test() {
  auto view(std::views::iota(1, 13));
  auto strided = std::views::stride(view, 3);
  assert(*strided.begin() == 1);
  assert(*(++strided.begin()) == 4);
  return true;
}

int main(int, char**) {
  static_assert(static_smoke_test());
  smoke_test();
  return 0;
}
