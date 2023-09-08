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

// basic_syncbuf(streambuf_type* obuf, const Allocator& allocator);
// basic_syncbuf(basic_syncbuf&& other);

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
public:
  test_syncbuf(test_buf<T>* buf, Alloc alloc) : std::basic_syncbuf<T, std::char_traits<T>, Alloc>(buf, alloc) {}

  void _setp(T* begin, T* end) { return this->setp(begin, end); }
};

template <class T>
struct test_allocator : std::allocator<T> {
  int id;
  test_allocator(int id = 0) : id(id) {}
};

template <class CharT>
void test_emit() {
  // We do this because we want to be able to any CharT
  CharT arr[3] = {'a', 'b', 'c'};
  CharT* ptr   = arr;

  test_buf<CharT> base;
  std::allocator<CharT> alloc;
  test_syncbuf<CharT> buff(&base, alloc);

  buff._setp(ptr, ptr + 3);
  assert(base._pptr() == nullptr);
  buff.emit();
  CharT* pptr = base._pptr();
  while (pptr) {
    assert(*pptr++ == *ptr++);
  }
}

template <class T>
void test_get_wrapped() {
  test_buf<T> base(42);
  std::allocator<T> alloc;
  test_syncbuf<T> buff(&base, alloc);
  assert(static_cast<test_buf<T>*>(buff.get_wrapped())->id == 42);
  ASSERT_NOEXCEPT(buff.get_wrapped());
}

template <class T>
void test_get_allocator() {
  test_buf<T> base;
  test_allocator<T> alloc(42);
  test_syncbuf<T, test_allocator<T>> buff(&base, alloc);
  assert(buff.get_allocator().id == 42);
  ASSERT_NOEXCEPT(buff.get_allocator());
}

template <class T>
void test_set_emit_on_sync() {
  // set_emit_on_sync tested in sync
  test_syncbuf<T, std::allocator<T>> buff(nullptr, std::allocator<T>());
  ASSERT_NOEXCEPT(buff.set_emit_on_sync(false));
}

int main(int, char**) {
  test_emit<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_emit<wchar_t>();
#endif

  test_get_wrapped<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_get_wrapped<wchar_t>();
#endif

  test_get_allocator<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_get_allocator<wchar_t>();
#endif

  test_set_emit_on_sync<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_set_emit_on_sync<wchar_t>();
#endif

  return 0;
}
