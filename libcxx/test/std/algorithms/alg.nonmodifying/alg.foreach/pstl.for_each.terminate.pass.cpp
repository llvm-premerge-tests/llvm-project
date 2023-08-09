//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <algorithm>

#include "check_assertion.h"
#include "test_execution_policies.h"

int main(int, char**) {
  test_execution_policies([](auto&& policy) {
    int a[] = {1, 2};
    EXPECT_STD_TERMINATE([&] { std::for_each(policy, std::begin(a), std::end(a), [](int) { throw int{}; }); });
    EXPECT_STD_TERMINATE([&] { std::for_each_n(policy, std::data(a), std::size(a), [](int) { throw int{}; }); });
  });
}
