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

template <typename T, typename SimdAbi, std::size_t array_size>
struct BroadCastHelper {
  const std::array<T, array_size>& expected_value;

  BroadCastHelper(const std::array<T, array_size>& value) : expected_value(value) {}

  template <typename U>
  void operator()() const {
    if constexpr (is_value_preserving_convertible<U, T>::value) {
      ex::simd<T, SimdAbi> simd_broadcast_from_vectorizable_type(static_cast<U>(3));
      assert_simd_value_correct<array_size>(simd_broadcast_from_vectorizable_type, expected_value);
    }
  }
};

template <class T, std::size_t>
struct CheckSimdBroadcastCtorFromVectorizedType {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (std::size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(3);

    types::for_each(arithmetic_no_bool_types(), BroadCastHelper<T, SimdAbi, array_size>(expected_value));
  }
};

template <typename T>
class implicit_type {
  T val;

public:
  implicit_type(T v) : val(v) {}
  operator T() const { return val; }
};

template <class T, std::size_t>
struct CheckSimdBroadcastCtor {
  template <class SimdAbi>
  void operator()() {
    constexpr std::size_t array_size = ex::simd_size_v<T, SimdAbi>;
    std::array<T, array_size> expected_value;
    for (std::size_t i = 0; i < array_size; ++i)
      expected_value[i] = static_cast<T>(3);

    implicit_type<T> implicit_convert_to_3(3);
    ex::simd<T, SimdAbi> simd_broadcast_from_implicit_type(std::move(implicit_convert_to_3));
    assert_simd_value_correct<array_size>(simd_broadcast_from_implicit_type, expected_value);

    int int_value_3 = 3;
    ex::simd<T, SimdAbi> simd_broadcast_from_int(std::move(int_value_3));
    assert_simd_value_correct<array_size>(simd_broadcast_from_int, expected_value);

    if constexpr (std::is_unsigned_v<T>) {
      unsigned int uint_value_3 = static_cast<unsigned int>(3);
      ex::simd<T, SimdAbi> simd_broadcast_from_uint(std::move(uint_value_3));
      assert_simd_value_correct<array_size>(simd_broadcast_from_uint, expected_value);
    }
  }
};

int main(int, char**) {
  test_all_simd_abi<CheckSimdBroadcastCtorFromVectorizedType>();
  test_all_simd_abi<CheckSimdBroadcastCtor>();
  return 0;
}
