//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class InputIterator>
//   basic_string(InputIterator begin, InputIterator end,
//   const Allocator& a = Allocator()); // constexpr since C++20

#include <string>
#include <iterator>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class It>
TEST_CONSTEXPR_CXX20 void test(It first, It last) {
  typedef typename std::iterator_traits<It>::value_type charT;
  typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
  typedef typename S::allocator_type A;
  S s2(first, last);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(std::distance(first, last)));
  unsigned i = 0;
  for (It it = first; it != last;) {
    assert(s2[i] == *it);
    ++it;
    ++i;
  }
  assert(s2.get_allocator() == A());
  assert(s2.capacity() >= s2.size());
}

template <class It, class A>
TEST_CONSTEXPR_CXX20 void test(It first, It last, const A& a) {
  typedef typename std::iterator_traits<It>::value_type charT;
  typedef std::basic_string<charT, std::char_traits<charT>, A> S;
  S s2(first, last, a);
  LIBCPP_ASSERT(s2.__invariants());
  assert(s2.size() == static_cast<std::size_t>(std::distance(first, last)));
  unsigned i = 0;
  for (It it = first; it != last;) {
    assert(s2[i] == *it);
    ++it;
    ++i;
  }
  assert(s2.get_allocator() == a);
  assert(s2.capacity() >= s2.size());
}

template <class Alloc>
TEST_CONSTEXPR_CXX20 void test(const Alloc& a) {
  const char* s = "12345678901234567890123456789012345678901234567890";

  test(s, s);
  test(s, s, Alloc(a));

  test(s, s + 1);
  test(s, s + 1, Alloc(a));

  test(s, s + 10);
  test(s, s + 10, Alloc(a));

  test(s, s + 50);
  test(s, s + 50, Alloc(a));

  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s));
  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s), Alloc(a));

  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1));
  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 1), Alloc(a));

  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10));
  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 10), Alloc(a));

  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 50));
  test(cpp17_input_iterator<const char*>(s), cpp17_input_iterator<const char*>(s + 50), Alloc(a));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test<test_allocator<char> >(test_allocator<char>(2));
#if TEST_STD_VER >= 11
  test<min_allocator<char> >(min_allocator<char>());
#endif
  {
    static_assert((!std::is_constructible<std::string, std::string, std::string>::value), "");
    static_assert((!std::is_constructible<std::string, std::string, std::string, std::allocator<char> >::value), "");
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
