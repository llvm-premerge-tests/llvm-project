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

// constexpr auto size() requires sized_range<V>
// { return ranges::size(base_); }
// constexpr auto size() const requires sized_range<const V>
// { return ranges::size(base_); }

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

template <class T>
concept HasSize = requires(T t) { t.size(); };

static_assert(HasSize<std::ranges::enumerate_view<SizedRange>>);
static_assert(HasSize<const std::ranges::enumerate_view<SizedRange>>);

static_assert(std::ranges::sized_range<Range>);

struct NonSizedRange : std::ranges::view_base {
  using iterator = forward_iterator<std::tuple<int>*>;
  iterator begin() const;
  iterator end() const;
};

static_assert(!std::ranges::sized_range<NonSizedRange>);
static_assert(!std::ranges::sized_range<const NonSizedRange>);

static_assert(!HasSize<std::ranges::enumerate_view<NonSizedRange>>);
static_assert(!HasSize<const std::ranges::enumerate_view<NonSizedRange>>);

constexpr bool test() {
  std::tuple<int> buffer[] = {1, 2, 3};

  // Non-const and const are sized
  {
    auto view = std::views::enumerate(buffer);
    assert(view.size() == 3);
    assert(std::as_const(view).size() == 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
