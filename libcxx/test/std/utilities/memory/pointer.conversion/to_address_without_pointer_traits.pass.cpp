//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;
//     Should not require a specialization of pointer_traits for Ptr.

#include <memory>

struct IntPtr {
  constexpr int* operator->() const { return ptr; }

  int* ptr;
};

template <typename T, bool>
struct TemplatedPtr {
  constexpr T* operator->() const { return ptr; }

  T* ptr;
};

constexpr bool test() {
  int i = 0;

  static_assert(std::to_address(IntPtr(nullptr)) == nullptr);
  static_assert(std::to_address(IntPtr(&i)) == &i);

  bool b = false;

  static_assert(std::to_address(TemplatedPtr<bool, true>(nullptr)) == nullptr);
  static_assert(std::to_address(TemplatedPtr<bool, true>(&b)) == &b);

  return true;
}

int main(int, char**) {
  static_assert(test());
  return 0;
}
