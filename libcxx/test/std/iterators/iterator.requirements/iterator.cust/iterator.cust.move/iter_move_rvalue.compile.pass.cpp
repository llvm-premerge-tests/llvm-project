//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test the constraints in libc++.

// template<class I>
// unspecified iter_move;

#include <array>
#include <utility>
#include <iterator>

template <typename I>
class wrapper {
public:
  wrapper() = default;

  constexpr explicit wrapper(I i) noexcept : base_(std::move(i)) {}

  // This operator is used to check the constraints work.
  constexpr decltype(auto) operator*() & noexcept { return base_; }

  // This operator is used to check the constraints work.
  constexpr auto&& operator*() && noexcept = delete;

private:
  I base_ = I{};
};

template <typename I>
wrapper(I) -> wrapper<I>;

void test_iter_move_rvalue() {
  std::array<int, 1> t1;
  auto first = wrapper{std::move(t1)};

  auto helper = [](auto&& f) requires (!requires{std::ranges::iter_move(std::move(f));}) {};
  [[maybe_unused]] auto dummy = std::ranges::iter_move(first);
  helper(first);
}
