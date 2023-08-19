//===------- BitFieldReflection.h - BitField Reflection Utils ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines simple reflection utility for bit-fields.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_SUPPORT_BIT_FIELD_REFLECTION_H
#define CLANG_SUPPORT_BIT_FIELD_REFLECTION_H

#include <cstddef>
#include <limits>
#include <tuple>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbitfield-constant-conversion"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#elif defined __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Woverflow"
#endif

namespace clang {

class BitFieldReflector {
public:
  template <typename FieldT>
  static constexpr auto countBits(FieldT F, size_t StartBit = 0) ->
      typename std::enable_if<std::is_unsigned<FieldT>::value, size_t>::type {
    return (StartBit == sizeof(FieldT) * 8)
               ? 0
               : (countBits(F, StartBit + 1) +
                  ((F & (1ULL << StartBit)) ? 1 : 0));
  }

  template <typename StructT, typename FieldT>
  static constexpr StructT instanceOf() {
    static_assert(std::is_unsigned<FieldT>::value,
                  "Bit field is not unsigned!");
    return instanceOfImpl<StructT, FieldT>(StructT{}, 0);
  }

private:
  template <typename StructT, typename FieldT, FieldT...>
  static constexpr StructT instanceOfImpl(StructT S, ...) {
    return S;
  }

  template <typename StructT, typename FieldT, FieldT... Inits>
  static constexpr StructT instanceOfImpl(StructT S, decltype(StructT{Inits...},
                                                              int()) = 0) {
    return instanceOfImpl<StructT, FieldT, Inits..., static_cast<FieldT>(-1)>(
        StructT{Inits...}, 0);
  }

private:
  struct is_implicitly_convertible {
    template <typename T> constexpr operator T() const;
  };

  template <typename, typename, typename...>
  struct is_initializable_ : std::false_type {};

  template <typename T, typename... From>
  struct is_initializable_<std::void_t<decltype(T{{std::declval<From>()}...})>,
                           T, From...> : std::true_type {};

  template <typename T, typename... From>
  using is_initializable = is_initializable_<std::void_t<>, T, From...>;

  template <typename, typename... Args>
  static constexpr size_t countFields(std::false_type) {
    return sizeof...(Args) ? sizeof...(Args) - 1 : 0;
  }

  template <typename T, typename... Args>
  static constexpr size_t countFields(std::true_type) {
    return countFields<T, Args..., is_implicitly_convertible>(
        is_initializable<T, Args..., is_implicitly_convertible>());
  }

public:
  template <typename T>
  struct FieldCounter
      : public std::integral_constant<size_t,
                                      countFields<T>(is_initializable<T>())> {};
};

} // end namespace clang

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // CLANG_SUPPORT_BIT_FIELD_REFLECTION_H
