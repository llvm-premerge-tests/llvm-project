//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <new>

#include "include/atomic_support.h"

#if defined(_LIBCPP_ABI_MICROSOFT)
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#elif defined(LIBCXX_BUILDING_LIBCXXABI)
// nothing to do, we use the one from libc++abi
#elif defined(LIBCXXRT)
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#elif defined(__GLIBCXX__)
// nothing to do, we use the one from libstdc++/libsupc++
#else
#  define _LIBPCPP_DEFINE_NEW_HANDLER
#endif

#if defined(_LIBPCPP_DEFINE_NEW_HANDLER)

#  if defined(_LIBCPP_ABI_MICROSOFT)

using _new_handler = int(__cdecl*)(size_t);
extern "C" _new_handler __cdecl _set_new_handler(_new_handler);

#  endif // _LIBCPP_ABI_MICROSOFT

namespace std { // purposefully not versioned

#  if defined(_LIBCPP_ABI_MICROSOFT)

static constinit new_handler __new_handler = nullptr;
static mutex __new_handler_lock;

int __cdecl _new_handler_interface(size_t) {
  __new_handler();
  return 1;
}

new_handler set_new_handler(new_handler handler) noexcept {
  lock_guard<mutex> guard(__new_handler_lock);
  new_handler old = __new_handler;
  __new_handler   = handler;
  _set_new_handler(handler ? _new_handler_interface : nullptr);
  return old;
}

new_handler get_new_handler() noexcept {
  lock_guard<mutex> guard(__new_handler_lock);
  return __new_handler;
}

#  else // _LIBCPP_ABI_MICROSOFT

static constinit std::new_handler __new_handler = nullptr;

new_handler set_new_handler(new_handler handler) noexcept { return __libcpp_atomic_exchange(&__new_handler, handler); }

new_handler get_new_handler() noexcept { return __libcpp_atomic_load(&__new_handler); }

#  endif // _LIBCPP_ABI_MICROSOFT

} // namespace std

#endif // _LIBPCPP_DEFINE_NEW_HANDLER
