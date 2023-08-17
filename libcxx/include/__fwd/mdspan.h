// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___MDSPAN_LAYOUTS_H
#define _LIBCPP___MDSPAN_LAYOUTS_H

#include <__config>
#include <array>
#include <span>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

// Layout policy with a mapping which corresponds to FORTRAN-style array layouts
struct layout_left {
  template <class _Extents>
  class mapping;
};

// Layout policy with a mapping which corresponds to C-style array layouts
struct layout_right {
  template <class _Extents>
  class mapping;
};

// Layout policy with a unique mapping where strides are arbitrary
struct layout_stride {
  template <class _Extents>
  class mapping;

// need this explicit deduction guide to silence warning ctad-maybe-unsupported
// LIBCPP_CTAD_SUPPORTED_FOR_TYPE does not appear to work for nested classes with
// out of class definition
#  ifdef _LIBCPP_COMPILER_CLANG_BASED
  template <class _Extents, class _OtherIndexType>
  [[maybe_unused]] mapping(_Extents, array<_OtherIndexType, _Extents::rank()>) -> mapping<_Extents>;

  template <class _Extents, class _OtherIndexType>
  [[maybe_unused]] mapping(_Extents, span<_OtherIndexType, _Extents::rank()>) -> mapping<_Extents>;
#  endif
};

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MDSPAN_LAYOUTS_H
