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
  template <typename T>
  static constexpr auto countBits(T Field, size_t StartBit = 0) ->
      typename std::enable_if<std::is_unsigned<T>::value, size_t>::type {
    return (StartBit == sizeof(T) * 8)
               ? 0
               : countBits(Field, StartBit + 1) +
                     ((Field & (1ull << StartBit)) ? 1 : 0);
  }

  template <typename StructT, typename FieldT>
  static constexpr StructT instanceOf() {
    using UnderlyingFieldType =
        typename std::conditional<std::is_enum<FieldT>::value,
                                  std::underlying_type<FieldT>,
                                  std::remove_cv<FieldT>>::type::type;
    static_assert(std::is_unsigned<UnderlyingFieldType>::value,
                  "Bit field is not unsigned!");
    return instanceOfImpl<StructT, FieldT>(StructT{}, 0);
  }

private:
  template <typename StructT, typename FieldT, FieldT... Inits>
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
  template <typename... Ts> struct make_void { typedef void type; };
  template <typename... Ts> using void_t = typename make_void<Ts...>::type;

  struct is_implicitly_convertible {
    template <typename Any> constexpr operator Any() const noexcept;
  };

  template <typename, typename T, typename...>
  struct is_aggregate_initializable_ : std::false_type {};

  template <typename T, typename... From>
  struct is_aggregate_initializable_<
      void_t<decltype(T{{std::declval<From>()}...})>, T, From...>
      : std::true_type {};

  template <typename T, typename... From>
  using is_aggregate_initializable =
      is_aggregate_initializable_<void_t<>, T, From...>;

  template <typename T, typename... Args>
  static constexpr size_t countFields(std::false_type) {
    return sizeof...(Args) ? sizeof...(Args) - 1 : 0;
  }

  template <typename T, typename... Args>
  static constexpr size_t countFields(std::true_type) {
    return countFields<T, Args..., is_implicitly_convertible>(
        is_aggregate_initializable<T, Args..., is_implicitly_convertible>());
  }

public:
  template <typename T>
  struct FieldCounter
      : public std::integral_constant<
            size_t, countFields<T>(is_aggregate_initializable<T>())> {};
};

} // end namespace clang

#ifdef __clang__
#pragma clang diagnostic pop
#elif defined __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // CLANG_SUPPORT_BIT_FIELD_REFLECTION_H
