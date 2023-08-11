//===-- Standalone implementation of std::optional --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc {
namespace cpp {

// Trivial in_place_t struct.
struct in_place_t {
  LIBC_INLINE constexpr explicit in_place_t() = default;
};

// Trivial nullopt_t struct.
struct nullopt_t {
  LIBC_INLINE constexpr explicit nullopt_t() = default;
};

// nullopt that can be used and returned.
LIBC_INLINE_VAR constexpr nullopt_t nullopt{};

// in_place that can be used in the constructor.
LIBC_INLINE_VAR constexpr in_place_t in_place{};

template <typename U, bool = !is_trivially_destructible<U>::value>
struct Storage {
  union {
    char empty;
    U stored_value;
  };

  LIBC_INLINE constexpr Storage() : empty() {}
  template <typename... Args>
  LIBC_INLINE constexpr explicit Storage(in_place_t, Args &&...args)
      : stored_value(forward<Args>(args)...) {}

  LIBC_INLINE ~Storage() { this->stored_value.~U(); }
};

template <typename U> struct Storage<U, false> {
  union {
    char empty;
    U stored_value;
  };

  LIBC_INLINE constexpr Storage() : empty() {}
  template <typename... Args>
  LIBC_INLINE constexpr explicit Storage(in_place_t, Args &&...args)
      : stored_value(forward<Args>(args)...) {}
};

template <typename T> struct OptionalStorageBase {
  Storage<T> storage;
  bool in_use = false;

  LIBC_INLINE ~OptionalStorageBase() = default;
  LIBC_INLINE constexpr OptionalStorageBase() = default;

  template <typename... ArgTypes>
  LIBC_INLINE constexpr OptionalStorageBase(in_place_t, ArgTypes &&...Args)
      : storage(in_place, forward<ArgTypes>(Args)...), in_use(true) {}

  LIBC_INLINE constexpr void reset() {
    if (in_use)
      storage.stored_value.~T();
    in_use = false;
  }

  LIBC_INLINE constexpr bool has_value() const { return in_use; }

  LIBC_INLINE constexpr T &value() & { return this->storage.stored_value; }
  LIBC_INLINE constexpr T const &value() const & {
    return this->storage.stored_value;
  }
  LIBC_INLINE constexpr T &&value() && {
    return move(this->storage.stored_value);
  }
};

// Class template that manages the storage for optionals.
template <typename T,
          bool /*_HasTrivialDestructor*/ = !is_trivially_destructible<T>::value>
struct OptionalStorage : OptionalStorageBase<T> {
  using OptionalStorageBase<T>::OptionalStorageBase;
  LIBC_INLINE constexpr OptionalStorage() = default;

  // Destructor needs to destroy the contained value:
  LIBC_INLINE ~OptionalStorage() { this->reset(); }
};

// storage for optionals with trivial destructors.
template <typename T>
struct OptionalStorage<T, false> : OptionalStorageBase<T> {
  using OptionalStorageBase<T>::OptionalStorageBase;
};

// This is very simple implementation of the std::optional class. It makes
// several assumptions that the underlying type is trivially constructable,
// copyable, or movable.
template <typename T> class optional {
  OptionalStorage<T> storage;

public:
  LIBC_INLINE constexpr optional() = default;
  LIBC_INLINE constexpr optional(nullopt_t) {}

  LIBC_INLINE constexpr optional(const T &t) : storage(in_place, t) {}
  LIBC_INLINE constexpr optional(const optional &) = default;

  LIBC_INLINE constexpr optional(T &&t) : storage(in_place, move(t)) {}
  LIBC_INLINE constexpr optional(optional &&O) = default;

  template <typename... ArgTypes>
  LIBC_INLINE constexpr optional(in_place_t, ArgTypes &&...Args)
      : storage(in_place, forward<ArgTypes>(Args)...) {}

  LIBC_INLINE constexpr optional &operator=(T &&t) {
    storage = move(t);
    return *this;
  }
  LIBC_INLINE constexpr optional &operator=(optional &&) = default;

  LIBC_INLINE constexpr optional &operator=(const T &t) {
    storage = t;
    return *this;
  }
  LIBC_INLINE constexpr optional &operator=(const optional &) = default;

  LIBC_INLINE constexpr void reset() { storage.reset(); }

  LIBC_INLINE constexpr const T &value() const & { return storage.value(); }

  LIBC_INLINE constexpr T &value() & { return storage.value(); }

  LIBC_INLINE constexpr explicit operator bool() const { return has_value(); }
  LIBC_INLINE constexpr bool has_value() const { return storage.has_value(); }
  LIBC_INLINE constexpr const T *operator->() const { return &storage.value(); }
  LIBC_INLINE constexpr T *operator->() { return &storage.value(); }
  LIBC_INLINE constexpr const T &operator*() const & { return value(); }
  LIBC_INLINE constexpr T &operator*() & { return value(); }

  LIBC_INLINE constexpr T &&value() && { return move(storage.value()); }
  LIBC_INLINE constexpr T &&operator*() && { return move(storage.value()); }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_OPTIONAL_H
