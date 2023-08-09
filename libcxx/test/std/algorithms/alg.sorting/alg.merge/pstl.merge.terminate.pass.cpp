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
#include "test_iterators.h"

int main(int, char**) {
  test_execution_policies([](auto&& policy) {
    EXPECT_STD_TERMINATE([&] {
      int a[] = {1, 2};
      std::merge(policy, std::begin(a), std::end(a), std::begin(a), std::end(a), std::begin(a), [](int, int) -> bool {
        throw int{};
      });
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::merge(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            util::throw_on_move_iterator(std::begin(a), 1),
            std::less{});
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate();
    });
  });

  return 0;
}
