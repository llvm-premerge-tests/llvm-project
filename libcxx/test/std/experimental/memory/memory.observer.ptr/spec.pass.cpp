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

int main(int, char**) {
  {
    std::experimental::observer_ptr<int> ptr1(new int(42));
    std::experimental::observer_ptr<int> ptr2(new int(101));
    std::experimental::swap(ptr1, ptr2);

    assert(*ptr1 == 101);
    assert(*ptr2 == 42);

    delete ptr1.get();
    delete ptr2.get();
  }
  {
    int* raw                                 = new int;
    std::experimental::observer_ptr<int> ptr = std::experimental::make_observer(raw);
    assert(ptr.get() == raw);
    delete raw;
  }

  std::experimental::observer_ptr<int> ptr1(new int);
  std::experimental::observer_ptr<int> ptr2(new int);
  std::experimental::observer_ptr<int> ptr3;
  std::experimental::observer_ptr<void> vptr((void*)new int);
  {
    assert(ptr1 == ptr1);
    assert(!(ptr1 == ptr2));
  }
  { assert(ptr1 != ptr2); }
  {
    assert(ptr3 == nullptr);
    assert(nullptr == ptr3);
    assert(!(ptr1 == nullptr));
    assert(!(nullptr == ptr1));
  }
  {
    assert(!(ptr3 != nullptr));
    assert(!(nullptr != ptr3));
    assert(ptr1 != nullptr);
    assert(nullptr != ptr1);
  }
  {
    assert((ptr1 < ptr1) == (ptr1.get() < ptr1.get()));
    assert((ptr1 < vptr) == (ptr1.get() < vptr.get()));
  }
  {
    assert((ptr1 > ptr1) == (ptr1.get() > ptr1.get()));
    assert((ptr1 > vptr) == (ptr1.get() > vptr.get()));
  }
  {
    assert((ptr1 <= ptr1) != (ptr1.get() > ptr1.get()));
    assert((ptr1 <= vptr) != (ptr1.get() > vptr.get()));
  }
  {
    assert((ptr1 >= ptr1) != (ptr1.get() < ptr1.get()));
    assert((ptr1 >= vptr) != (ptr1.get() < vptr.get()));
  }

  delete ptr1.get();
  delete ptr2.get();
  delete (int*)vptr.get();

  return 0;
}