//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string(basic_string<charT,traits,Allocator>&& str); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s0) {
  S s1 = s0;
  S s2 = std::move(s0);
  LIBCPP_ASSERT(s2.__invariants());
  LIBCPP_ASSERT(s0.__invariants());
  assert(s2 == s1);
  assert(s2.capacity() >= s2.size());
  assert(s2.get_allocator() == s1.get_allocator());
}

template <class CharT, template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void test_string() {
  using A = Alloc<CharT>;
  using S = std::basic_string<CharT, std::char_traits<CharT>, A>;
  test(S(A{}));
  test(S("1", A()));
  test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A()));
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    using A = test_allocator<char>;
    using S = std::basic_string<char, std::char_traits<char>, A>;
    test(S(A(3)));
    test(S("1", A(5)));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)));
  }
  test_string<char, min_allocator>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
