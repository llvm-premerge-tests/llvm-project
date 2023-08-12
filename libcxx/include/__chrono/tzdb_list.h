// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_TZDB_LIST_H
#define _LIBCPP___CHRONO_TZDB_LIST_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#  include <__availability>
#  include <__chrono/tzdb.h>
#  include <__mutex/unique_lock.h>
#  include <forward_list>
#  include <string_view>

// When threads are not available the locking is not required.
#  ifndef _LIBCPP_HAS_NO_THREADS
#    include <shared_mutex>
#  endif

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20

namespace chrono {

// TODO TZDB Document and test _LIBCPP_NODISCARD_EXT.
class _LIBCPP_AVAILABILITY_TZDB tzdb_list {
public:
  _LIBCPP_HIDE_FROM_ABI explicit tzdb_list(tzdb&& __tzdb) { __tzdb_.push_front(std::move(__tzdb)); }

  tzdb_list(const tzdb_list&)            = delete;
  tzdb_list& operator=(const tzdb_list&) = delete;

  using const_iterator = forward_list<tzdb>::const_iterator;

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI const tzdb& front() const noexcept {
#    ifndef _LIBCPP_HAS_NO_THREADS
    shared_lock __lock{__mutex_};
#    endif
    return __tzdb_.front();
  }

  _LIBCPP_HIDE_FROM_ABI const_iterator erase_after(const_iterator __p) {
#    ifndef _LIBCPP_HAS_NO_THREADS
    unique_lock __lock{__mutex_};
#    endif
    return __tzdb_.erase_after(__p);
  }

  _LIBCPP_HIDE_FROM_ABI tzdb& __emplace_front(tzdb&& __tzdb) {
#    ifndef _LIBCPP_HAS_NO_THREADS
    unique_lock __lock{__mutex_};
#    endif
    return __tzdb_.emplace_front(std::move(__tzdb));
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI const_iterator begin() const noexcept {
#    ifndef _LIBCPP_HAS_NO_THREADS
    shared_lock __lock{__mutex_};
#    endif
    return __tzdb_.begin();
  }
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI const_iterator end() const noexcept {
    //  forward_list<T>::end does not access the list.
    return __tzdb_.end();
  }

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI const_iterator cbegin() const noexcept { return begin(); }
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI const_iterator cend() const noexcept { return end(); }

private:
#    ifndef _LIBCPP_HAS_NO_THREADS
  mutable shared_mutex __mutex_;
#    endif
  forward_list<tzdb> __tzdb_;
};

// This function allows users to override the directory where the TZDB
// files are stored. This should rarely be needed, a better solution
// would be to set the proper value at build time in
// _LIBCPP_CHRONO_TZDB_DIRECTORY
_LIBCPP_AVAILABILITY_TZDB _LIBCPP_OVERRIDABLE_FUNC_VIS string_view __libcpp_tzdb_directory();

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI tzdb_list& get_tzdb_list();

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_HIDE_FROM_ABI inline const tzdb& get_tzdb() {
  return get_tzdb_list().front();
}

_LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI const tzdb& reload_tzdb();

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_EXPORTED_FROM_ABI string remote_version();

} // namespace chrono

#  endif //_LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#endif // _LIBCPP___CHRONO_TZDB_LIST_H
