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

struct ThrowOnCompare {};

#ifndef TEST_HAS_NO_EXCEPTIONS
bool operator==(ThrowOnCompare, ThrowOnCompare) { throw int{}; }
#endif

int main(int, char**) {
  test_execution_policies([&](auto&& policy) {
    EXPECT_STD_TERMINATE([&] {
      ThrowOnCompare a[2]{};
      (void)std::replace(policy, std::begin(a), std::end(a), ThrowOnCompare{}, ThrowOnCompare{});
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::replace(
            policy, util::throw_on_move_iterator(std::begin(a), 1), util::throw_on_move_iterator(std::end(a), 1), 1, 2);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate();
    });

    EXPECT_STD_TERMINATE([&] {
      ThrowOnCompare a[2]{};
      (void)std::replace_if(
          policy, std::begin(a), std::end(a), [](ThrowOnCompare&) -> bool { throw int{}; }, ThrowOnCompare{});
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::replace_if(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            [](int) { return true; },
            2);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate();
    });

    EXPECT_STD_TERMINATE([&] {
      ThrowOnCompare a[2]{};
      (void)std::replace_copy(policy, std::begin(a), std::end(a), std::begin(a), ThrowOnCompare{}, ThrowOnCompare{});
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::replace_copy(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            util::throw_on_move_iterator(std::begin(a), 1),
            1,
            2);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate();
    });

    EXPECT_STD_TERMINATE([&] {
      ThrowOnCompare a[2]{};
      (void)std::replace_copy_if(
          policy,
          std::begin(a),
          std::end(a),
          std::begin(a),
          [](ThrowOnCompare& i) { return i == i; },
          ThrowOnCompare{});
    });
    EXPECT_STD_TERMINATE([&] {
      try {
        int a[] = {1, 2};
        (void)std::replace_copy_if(
            policy,
            util::throw_on_move_iterator(std::begin(a), 1),
            util::throw_on_move_iterator(std::end(a), 1),
            util::throw_on_move_iterator(std::begin(a), 1),
            [](int) { return true; },
            2);
      } catch (const util::iterator_error&) {
        assert(false);
      }
      std::terminate();
    });
  });
}
