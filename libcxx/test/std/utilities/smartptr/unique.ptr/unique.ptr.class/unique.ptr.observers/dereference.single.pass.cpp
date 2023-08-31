//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op*()

#include <memory>
#include <cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 11
struct ThrowDereference {
  TEST_CONSTEXPR_CXX23 ThrowDereference& operator*() noexcept(false);
  TEST_CONSTEXPR_CXX23 operator bool() const { return false; }
};

struct Deleter {
  using pointer = ThrowDereference;
  TEST_CONSTEXPR_CXX23 void operator()(ThrowDereference&) const {}
};
#endif

TEST_CONSTEXPR_CXX23 bool test() {
  {
    std::unique_ptr<int> p(new int(3));
    assert(*p == 3);
    ASSERT_NOEXCEPT(*p);
  }
#if TEST_STD_VER >= 11
  {
    // The noexcept status of *unique_ptr<>::pointer should be propagated.
    std::unique_ptr<ThrowDereference, Deleter> p;
    ASSERT_NOT_NOEXCEPT(*p);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
