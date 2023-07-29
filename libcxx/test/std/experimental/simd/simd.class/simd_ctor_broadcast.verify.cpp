//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.class]
// template<class U> simd(U&& value) noexcept;

#include "../test_utils.h"

namespace ex = std::experimental::parallelism_v2;

template <typename T>
class no_implicit_val {
  T val;

public:
  no_implicit_val(T val) : val(val) {}
};

int main(int, char**) {
  ex::native_simd<int> s1(2.0f);
  // expected-error-re@* {{no matching constructor for initialization of {{.*}}}}

  no_implicit_val<int> value(3);
  ex::native_simd<int> s2(std::move(value));
  // expected-error-re@* {{no matching constructor for initialization of {{.*}}}}

  return 0;
}
