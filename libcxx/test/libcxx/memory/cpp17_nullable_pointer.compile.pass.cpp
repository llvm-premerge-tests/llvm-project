//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

#include <__iterator/cpp17_iterator_concepts.h>
#include <memory>
#include <cstddef>

// Types

struct non_equality_comparable_ptr {
  non_equality_comparable_ptr()                                                  = default;
  non_equality_comparable_ptr(const non_equality_comparable_ptr&) noexcept       = default;
  non_equality_comparable_ptr(non_equality_comparable_ptr&&) noexcept            = default;
  non_equality_comparable_ptr& operator=(non_equality_comparable_ptr&&) noexcept = default;
  bool operator==(const non_equality_comparable_ptr&) const noexcept             = delete;
};

struct non_default_constructable_ptr {
  non_default_constructable_ptr()                                                    = delete;
  non_default_constructable_ptr(const non_default_constructable_ptr&) noexcept       = default;
  non_default_constructable_ptr(non_default_constructable_ptr&&) noexcept            = default;
  non_default_constructable_ptr& operator=(non_default_constructable_ptr&&) noexcept = default;
  bool operator==(const non_default_constructable_ptr&) const noexcept               = default;
};

struct non_copy_constructable_ptr {
  non_copy_constructable_ptr()                                                 = default;
  non_copy_constructable_ptr(const non_copy_constructable_ptr&) noexcept       = delete;
  non_copy_constructable_ptr(non_copy_constructable_ptr&&) noexcept            = default;
  non_copy_constructable_ptr& operator=(non_copy_constructable_ptr&&) noexcept = default;
  bool operator==(const non_copy_constructable_ptr&) const noexcept            = default;
};

struct non_move_assignable_ptr {
  non_move_assignable_ptr()                                              = default;
  non_move_assignable_ptr(const non_move_assignable_ptr&) noexcept       = default;
  non_move_assignable_ptr(non_move_assignable_ptr&&) noexcept            = default;
  non_move_assignable_ptr& operator=(non_move_assignable_ptr&&) noexcept = delete;
  bool operator==(const non_move_assignable_ptr&) const noexcept         = default;
};

class Nullable {
  int id = 0;

public:
  Nullable() = default;
  Nullable(std::nullptr_t) {}
  explicit operator bool() const { return id != 0; }
  friend bool operator==(Nullable l, Nullable r) { return l.id == r.id; }
};

// Tests

static_assert(std::__cpp17_equality_comparable<Nullable>);
static_assert(std::__cpp17_equality_comparable<std::shared_ptr<int>>);
static_assert(!std::__cpp17_equality_comparable<non_equality_comparable_ptr>);
static_assert(std::__cpp17_equality_comparable<non_default_constructable_ptr>);
static_assert(std::__cpp17_equality_comparable<non_copy_constructable_ptr>);
static_assert(std::__cpp17_equality_comparable<non_move_assignable_ptr>);

static_assert(std::__cpp17_default_constructible<Nullable>);
static_assert(std::__cpp17_default_constructible<std::shared_ptr<int>>);
static_assert(std::__cpp17_default_constructible<non_equality_comparable_ptr>);
static_assert(!std::__cpp17_default_constructible<non_default_constructable_ptr>);
static_assert(std::__cpp17_default_constructible<non_copy_constructable_ptr>);
static_assert(std::__cpp17_default_constructible<non_move_assignable_ptr>);

static_assert(std::__cpp17_copy_constructible<Nullable>);
static_assert(std::__cpp17_copy_constructible<std::shared_ptr<int>>);
static_assert(std::__cpp17_copy_constructible<non_equality_comparable_ptr>);
static_assert(std::__cpp17_copy_constructible<non_default_constructable_ptr>);
static_assert(!std::__cpp17_copy_constructible<non_copy_constructable_ptr>);
static_assert(std::__cpp17_copy_constructible<non_move_assignable_ptr>);

static_assert(std::__cpp17_move_assignable<Nullable>);
static_assert(std::__cpp17_move_assignable<std::shared_ptr<int>>);
static_assert(std::__cpp17_move_assignable<non_equality_comparable_ptr>);
static_assert(std::__cpp17_move_assignable<non_default_constructable_ptr>);
static_assert(std::__cpp17_move_assignable<non_copy_constructable_ptr>);
static_assert(!std::__cpp17_move_assignable<non_move_assignable_ptr>);

static_assert(std::__cpp17_swappable<Nullable>);
static_assert(std::__cpp17_swappable<std::shared_ptr<int>>);
static_assert(std::__cpp17_swappable<non_equality_comparable_ptr>);
static_assert(std::__cpp17_swappable<non_default_constructable_ptr>);
static_assert(std::__cpp17_swappable<non_copy_constructable_ptr>);
static_assert(!std::__cpp17_swappable<non_move_assignable_ptr>);

static_assert(std::__cpp17_nullable_pointer<Nullable>);
static_assert(std::__cpp17_nullable_pointer<std::shared_ptr<int>>);
static_assert(!std::__cpp17_nullable_pointer<std::unique_ptr<int>>);
