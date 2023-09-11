//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14 ,c++17

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_osyncstream;

// basic_osyncstream& operator=(basic_osyncstream&& rhs);

#include <syncstream>
#include <cassert>

#include "test_macros.h"

template <class T>
struct test_allocator : std::allocator<T> {
  int id;
  int moves  = 0;
  int copies = 0;

  test_allocator(int _id = 0) : id(_id) {}
};

template <class T>
class test_buf : public std::basic_streambuf<T> {};

template <class CharT>
void test() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  test_buf<CharT> base;
  test_allocator<CharT> alloc(42);
  OS out1(&base, alloc);
  typename OS::syncbuf_type* buff1 = out1.rdbuf();

  assert(out1.get_wrapped() == &base);
  assert(buff1->get_wrapped() == &base);

  static_assert(!noexcept(out1.operator=(std::move(out1)))); // LWG-3867
  OS out2                          = std::move(out1);
  typename OS::syncbuf_type* buff2 = out2.rdbuf();

  assert(out2.get_wrapped() == &base);
  assert(buff2->get_wrapped() == &base);
  assert(buff2->get_allocator().id == alloc.id);
  assert(out1.get_wrapped() == nullptr);
  assert(buff1->get_wrapped() == nullptr);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
