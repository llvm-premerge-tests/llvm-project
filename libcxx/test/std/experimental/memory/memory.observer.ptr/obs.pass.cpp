// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: c++experimental

// <experimental/memory>

// observer_ptr

#include <experimental/memory>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct Foo {
  const int x;
  constexpr Foo(const int x) : x(x) {}
};

int main(int, char**) {
  static constexpr int x       = 42;
  constexpr const int* raw_ptr = &x;
  {
    constexpr std::experimental::observer_ptr<const int> ptr(raw_ptr);
    static_assert(ptr.get() == raw_ptr);
  }
  {
    constexpr std::experimental::observer_ptr<const int> ptr1;
    const bool check1 = ptr1;
    static_assert(check1 == false);

    constexpr std::experimental::observer_ptr<const int> ptr2(raw_ptr);
    const bool check2 = ptr2;
    static_assert(check2 == true);
  }
  {
    static constexpr Foo f       = Foo(42);
    constexpr const Foo* foo_ptr = &f;
    constexpr std::experimental::observer_ptr<const Foo> ptr(foo_ptr);
    static_assert(ptr->x == 42);
    static_assert((*ptr).x == 42);
  }

  return 0;
}