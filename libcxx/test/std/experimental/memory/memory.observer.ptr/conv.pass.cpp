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

template <class T>
void test_convertibility() {
  typedef std::experimental::observer_ptr<T> OP;
  static_assert(std::is_convertible<OP, T*>::value, "");
}

struct Foo;

struct Bar {};

int main(int, char**) {
  test_convertibility<void>();
  test_convertibility<bool>();
  test_convertibility<int>();
  test_convertibility<Foo>();
  test_convertibility<Bar>();

  return 0;
}