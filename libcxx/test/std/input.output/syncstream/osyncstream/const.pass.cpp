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

// basic_osyncstream(streambuf_type* buf, const Allocator& allocator);
// basic_osyncstream(basic_osyncstream&& other) noexcept;

#include <syncstream>
#include <sstream>
#include <cassert>
#include <iostream>

#include "test_macros.h"

template <class T>
struct test_allocator : std::allocator<T> {
  int id;
  int moves  = 0;
  int copies = 0;

  test_allocator(int id = 0) : id(id) {}
  test_allocator(test_allocator const& other) : id(other.id), moves(other.moves), copies(other.copies) {}
  test_allocator(test_allocator&& other)
      : id(std::move(other.id)), moves(std::move(other.moves)), copies(std::move(other.copies)) {}
};

template <class T>
class test_buf : public std::basic_streambuf<T> {};

template <class CharT>
void test_const() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  test_buf<CharT> base;
  test_allocator<CharT> alloc(42);
  OS out(&base, alloc);
  typename OS::syncbuf_type* buff = out.rdbuf();

  assert(out.get_wrapped() == &base);
  assert(buff->get_wrapped() == &base);
  assert(buff->get_allocator().id == alloc.id);
};

template <class CharT>
void test_rvalue() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  test_buf<CharT> base;
  test_allocator<CharT> alloc(42);
  OS out1(&base, alloc);
  typename OS::syncbuf_type* buff1 = out1.rdbuf();

  assert(out1.get_wrapped() == &base);
  assert(buff1->get_wrapped() == &base);

  OS out2(std::move(out1));
  typename OS::syncbuf_type* buff2 = out2.rdbuf();

  assert(out2.get_wrapped() == &base);
  assert(buff2->get_wrapped() == &base);
  assert(buff2->get_allocator().id == alloc.id);
  assert(out1.get_wrapped() == nullptr);
  assert(buff1->get_wrapped() == nullptr);
}

template <class CharT>
void test_dest() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;
  using SS = std::basic_ostringstream<CharT>;
  CharT c  = 'f';

  SS ss;
  {
    OS out(ss);
    out << c;
    assert(ss.str().empty());
  }
  assert(ss.str()[0] == c);
}

int main(int, char**) {
  test_const<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_const<wchar_t>();
#endif

  test_rvalue<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_rvalue<wchar_t>();
#endif

  test_dest<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_dest<wchar_t>();
#endif

  return 0;
}
