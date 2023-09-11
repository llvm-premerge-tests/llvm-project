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
#include <cassert>

#include "poisoned_hash_helper.h"
#include "test_macros.h"

struct A {};

int main(int, char**) {
  {
    int* ptr = new int;
    std::experimental::observer_ptr<int> p(ptr);
    std::hash<std::experimental::observer_ptr<int>> f;
    std::size_t h = f(p);
    assert(h == std::hash<int*>()(ptr));
  }
  {
    test_hash_enabled_for_type<std::experimental::observer_ptr<int>>();
    test_hash_enabled_for_type<std::experimental::observer_ptr<A>>();
  }

  return 0;
}
