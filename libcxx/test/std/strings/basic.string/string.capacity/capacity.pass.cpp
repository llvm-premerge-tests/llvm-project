//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// size_type capacity() const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_allocator.h"
#include "min_allocator.h"

#include "test_macros.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, test_allocator_statistics& alloc_stats) {
  alloc_stats.throw_after = 0;
#ifndef TEST_HAS_NO_EXCEPTIONS
  try
#endif
  {
    while (s.size() < s.capacity())
      s.push_back(typename S::value_type());
    assert(s.size() == s.capacity());
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  catch (...) {
    assert(false);
  }
#endif
  alloc_stats.throw_after = INT_MAX;
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  S s;
  assert(s.capacity() > 0);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    using S = std::basic_string<char, std::char_traits<char>, test_allocator<char> >;
    test_allocator_statistics alloc_stats;
    S s((test_allocator<char>(&alloc_stats)));
    test(s, alloc_stats);
    s.assign(10, 'a');
    s.erase(5);
    test(s, alloc_stats);
    s.assign(100, 'a');
    s.erase(50);
    test(s, alloc_stats);
  }
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
