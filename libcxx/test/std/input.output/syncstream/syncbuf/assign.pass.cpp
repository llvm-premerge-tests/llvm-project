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
// class basic_syncbuf;

// basic_syncbuf& operator=(basic_syncbuf&& rhs) noexcept;
// void swap(basic_syncbuf& other) noexcept;

#include <syncstream>
#include <cassert>

#include "test_macros.h"

template <class T>
class test_buf : public std::basic_streambuf<T> {
public:
  int id;

  test_buf(int id = 0) : id(id) {}

  T* _pptr() { return this->pptr(); }
};

template <class T, class Alloc = std::allocator<T>>
class test_syncbuf : public std::basic_syncbuf<T, std::char_traits<T>, Alloc> {
  using Base = std::basic_syncbuf<T, std::char_traits<T>, Alloc>;

public:
  test_syncbuf() = default;

  test_syncbuf(test_buf<T>* buf, Alloc alloc) : Base(buf, alloc) {}

  test_syncbuf(typename Base::streambuf_type* buf, Alloc alloc) : Base(buf, alloc) {}

  void _setp(T* begin, T* end) { return this->setp(begin, end); }
};

template <class CharT>
void test_emit_on_assign() {
  CharT arr[3] = {'a', 'b', 'c'};
  CharT* ptr   = arr;

  test_buf<CharT> base;
  std::allocator<CharT> alloc;
  test_syncbuf<CharT> buff(&base, alloc);

  buff._setp(ptr, ptr + 3);
  assert(base._pptr() == nullptr);
  buff = test_syncbuf<CharT>();
  assert(buff.get_wrapped() == nullptr);
  CharT* pptr = base._pptr();
  while (pptr) {
    assert(*pptr++ == *ptr++);
  }
}

template <class CharT>
void test_assign() {
  using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  test_syncbuf<CharT> base;
  std::allocator<CharT> alloc;
  BuffT buff1(&base, alloc);
  BuffT buff2 = std::move(buff1);

  assert(buff1.get_wrapped() == nullptr);
  assert(buff2.get_wrapped() == &base);

  test_emit_on_assign<CharT>();

  // Also LWG issue?
  // assert(buff1.rdbuf()->pbase() == buff1.rdbuf()->pptr());
}

template <class CharT>
void test_swap() {
  using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  test_syncbuf<CharT> base1;
  test_syncbuf<CharT> base2;
  std::allocator<CharT> alloc;
  BuffT buff1(&base1, alloc);
  BuffT buff2(&base2, alloc);
  buff1.swap(std::move(buff2));

  assert(buff1.get_wrapped() == &base2);
  assert(buff2.get_wrapped() == &base1);
}

int main(int, char**) {
  test_assign<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_assign<wchar_t>();
#endif

  test_swap<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_swap<wchar_t>();
#endif

  return 0;
}
