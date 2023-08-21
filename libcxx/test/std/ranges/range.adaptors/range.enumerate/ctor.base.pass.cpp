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

// constexpr explicit enumerate_view(V base);

#include <ranges>

#include <cassert>
#include <tuple>
#include <type_traits>

constexpr int buff[] = {0, 1};

struct DefaultConstructibleView : std::ranges::view_base {
  constexpr DefaultConstructibleView() : begin_(buff), end_(buff + 1) {}
  constexpr int const* begin() const { return begin_; }
  constexpr int const* end() const { return end_; }

  int const* begin_;
  int const* end_;
};

template <bool DefaultInitializable>
struct DefaultInitializableView : std::ranges::view_base {
  constexpr explicit DefaultInitializableView()
    requires DefaultInitializable
  = default;

  int* begin() const;
  int* end() const;
};

struct NoDefaultView : std::ranges::view_base {
  NoDefaultView() = delete;

  int* begin() const;
  int* end() const;
};

// SFINAE

static_assert(!std::is_default_constructible_v<std::ranges::enumerate_view<NoDefaultView>>);
static_assert(std::is_default_constructible_v<std::ranges::enumerate_view<DefaultInitializableView<true>>>);
static_assert(!std::is_default_constructible_v<std::ranges::enumerate_view<DefaultInitializableView<false>>>);

constexpr bool test() {
  using EnumerateView = std::ranges::enumerate_view<DefaultConstructibleView>;

  {
    EnumerateView view;

    assert((*view.begin() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{0, 0}));
    assert((*view.end() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{1, 1}));

    auto [bi, bv] = *view.begin();
    assert(bi == 0);
    assert(bv == 0);

    auto [ei, ev] = *view.end();
    assert(ei == 1);
    assert(ev == 1);
  }
  {
    EnumerateView view = {};

    assert((*view.begin() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{0, 0}));
    assert((*view.end() == std::tuple<std::ranges::iterator_t<EnumerateView>::difference_type, int>{1, 1}));

    auto [bi, bv] = *view.begin();
    assert(bi == 0);
    assert(bv == 0);

    auto [ei, ev] = *view.end();
    assert(ei == 1);
    assert(ev == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
