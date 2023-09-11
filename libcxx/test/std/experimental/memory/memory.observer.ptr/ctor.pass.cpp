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
void custom_delete(T* ptr) {
  delete ptr;
}
void custom_delete(void* ptr) { delete (int*)ptr; }

template <class T, class U>
void assert_constructability() {
  typedef std::experimental::observer_ptr<T> OP;
  static_assert(std::is_nothrow_constructible<OP>::value, "");
  static_assert(std::is_nothrow_constructible<OP, std::nullptr_t>::value, "");
  static_assert(std::is_nothrow_constructible<OP, T*>::value, "");
  static_assert(std::is_nothrow_constructible<OP, OP const&>::value, "");
  static_assert(std::is_nothrow_constructible<OP, OP&&>::value, "");
  static_assert(std::is_nothrow_constructible<OP, std::experimental::observer_ptr<U>>::value ==
                    std::is_convertible<U*, T*>::value,
                "");
}

template <class T>
void construct_nullptr() {
  {
    std::experimental::observer_ptr<T> ptr;
    assert(ptr.get() == nullptr);
  }
  {
    std::experimental::observer_ptr<T> ptr(nullptr);
    assert(ptr.get() == nullptr);
  }
}

template <class T>
void construct_ptr(T* raw_ptr) {
  std::experimental::observer_ptr<T> ptr(raw_ptr);
  assert(ptr.get() == raw_ptr);
  custom_delete(raw_ptr);
}

template <class T, class U>
void construct_other(U* raw_ptr) {
  std::experimental::observer_ptr<U> uptr(raw_ptr);
  std::experimental::observer_ptr<T> tptr(uptr);
  assert(uptr.get() == raw_ptr);
  assert(tptr.get() == raw_ptr);
  custom_delete(raw_ptr);
}

template <class T>
void test_copy_move(T* raw_ptr) {
  std::experimental::observer_ptr<T> ptr_a(raw_ptr);
  std::experimental::observer_ptr<T> ptr_b(ptr_a);
  std::experimental::observer_ptr<T> ptr_c(std::move(ptr_a));
  assert(ptr_b.get() == raw_ptr);
  assert(ptr_c.get() == raw_ptr);
  custom_delete(raw_ptr);
}

struct Foo;
struct Bar {
  Bar(int) {}
};

int main(int, char**) {
  {
    assert_constructability<void, void>();
    assert_constructability<void, int>();
    assert_constructability<void, Bar>();
    assert_constructability<Foo, void>();
    assert_constructability<Foo, int>();
    assert_constructability<Foo, Bar>();
    assert_constructability<int, void>();
    assert_constructability<int, int>();
    assert_constructability<int, Bar>();
  }
  {
    construct_nullptr<Foo>();
    construct_nullptr<Bar>();
    construct_nullptr<int>();
    construct_nullptr<void>();
  }
  {
    construct_ptr(new int);
    construct_ptr(new Bar(42));
    construct_ptr((void*)new int);
  }
  {
    construct_other<void>(new int);
    construct_other<void>(new Bar(42));

    // overload resolution
    typedef std::experimental::observer_ptr<int> OP1;
    typedef std::experimental::observer_ptr<Bar> OP2;
    typedef std::experimental::observer_ptr<char> OP3;
    static_assert(!std::is_nothrow_constructible<OP1, OP2>::value, "");
    static_assert(!std::is_nothrow_constructible<OP1, OP3>::value, "");
  }
  {
    test_copy_move(new int);
    test_copy_move((void*)new int);
    test_copy_move(new Bar(42));
  }

  return 0;
}