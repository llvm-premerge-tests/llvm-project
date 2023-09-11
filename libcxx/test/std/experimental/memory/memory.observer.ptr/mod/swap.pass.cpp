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

constexpr bool constexpr_swap() {
  int x = 42;
  int y = 55;

  auto op1 = std::experimental::observer_ptr<int>(&x);
  auto op2 = std::experimental::observer_ptr<int>(&y);

  op1.swap(op2);
  return *op1 == 55 && *op2 == 42;
}

int main(int, char**) {
  int* raw_ptr1 = new int;
  int* raw_ptr2 = new int;

  {
    std::experimental::observer_ptr<int> ptr1;
    std::experimental::observer_ptr<int> ptr2;
    ptr1.swap(ptr2);
    assert(ptr1.get() == nullptr);
    assert(ptr2.get() == nullptr);
  }
  {
    std::experimental::observer_ptr<int> ptr1(raw_ptr1);
    std::experimental::observer_ptr<int> ptr2;
    ptr1.swap(ptr2);
    assert(ptr1.get() == nullptr);
    assert(ptr2.get() == raw_ptr1);
  }
  {
    std::experimental::observer_ptr<int> ptr1;
    std::experimental::observer_ptr<int> ptr2(raw_ptr2);
    ptr1.swap(ptr2);
    assert(ptr1.get() == raw_ptr2);
    assert(ptr2.get() == nullptr);
  }
  {
    std::experimental::observer_ptr<int> ptr1(raw_ptr1);
    std::experimental::observer_ptr<int> ptr2(raw_ptr2);
    ptr1.swap(ptr2);
    assert(ptr1.get() == raw_ptr2);
    assert(ptr2.get() == raw_ptr1);
  }

  delete raw_ptr1;
  delete raw_ptr2;

  static_assert(constexpr_swap());

  return 0;
}