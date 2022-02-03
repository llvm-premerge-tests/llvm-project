//===----------------------------------------------------------------------===////
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===////

#ifndef _LIBCPP_INTERNAL_THREADING_SUPPORT_H
#define _LIBCPP_INTERNAL_THREADING_SUPPORT_H

/** internal_threading_support.h - Equivalent to __threading_support, but for internal uses that are
 * not related to the C++11 Threading Library. These should all be within code that is compiled into
 * the libcxx library rather than potentially #included by user code.
 *
 * See libcxx/docs/DesignDocs/InternalThreadSynchronization.rst for details.
 *
 * Any use of the C++11 Thread Support Library (e.g. std::mutex) without an underlying thread
 * library available is user error, so we want files like mutex.cpp to use the definitions from
 * __threading_support. For that reason, _LIBCPP_HAS_THREAD_LIBRARY_OPTIONAL needs to be defined
 * here instead of in __config.
 *
 * Any platforms which cannot determine during compilation of libc++ the availability of a thread
 * library such as pthreads should define `_LIBCPP_HAS_THREAD_LIBRARY_OPTIONAL` before
 * <__threading_support> is #included, and provide definitions for
 * `__libcpp_is_threading_api_enabled()` and `__libcpp_might_have_multiple_threads()`
 */

#if !defined(_LIBCPP_BUILDING_LIBRARY)
#  error internal_threading_support.h is only intended for internal use by libcxx
#endif

#ifdef _LIBCPP_THREADING_SUPPORT
#  error internal_threading_support.h must be included before <__threading_support>
#endif

#if defined(__MVS__)
// Tell __threading_support not to provide function definitions, since we provide them here
#  define _LIBCPP_HAS_THREAD_LIBRARY_OPTIONAL
#endif

#include <__threading_support>

#if defined(_LIBCPP_HAS_THREAD_LIBRARY_OPTIONAL) && !defined(_LIBCPP_HAS_THREAD_LIBRARY_EXTERNAL)

_LIBCPP_THREAD_ABI_VISIBILITY
bool __libcpp_is_threading_api_enabled();

_LIBCPP_THREAD_ABI_VISIBILITY
bool __libcpp_might_have_multiple_threads();

#  if defined(__MVS__)
#    include "ceeedb.h"
/// On z/OS some posix functions can be enabled/disabled at runtime.
/// However, the enabled/disabled status should not change over the life of the process.
bool __libcpp_is_threading_api_enabled() {
  static bool __posix_on = __libcpp_ceeedb_posix();
  return __posix_on;
}
bool __libcpp_might_have_multiple_threads() { return __libcpp_ceeedb_multithread(); }
#  endif

//===----------------------------------------------------------------------===//
//    Definitions which would normally be provided by __threading_support
//===----------------------------------------------------------------------===//

#  if defined(_LIBCPP_HAS_THREAD_API_PTHREAD)

int __libcpp_recursive_mutex_init(__libcpp_recursive_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;

  pthread_mutexattr_t attr;
  int __ec = pthread_mutexattr_init(&attr);
  if (__ec)
    return __ec;
  __ec = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  if (__ec) {
    pthread_mutexattr_destroy(&attr);
    return __ec;
  }
  __ec = pthread_mutex_init(__m, &attr);
  if (__ec) {
    pthread_mutexattr_destroy(&attr);
    return __ec;
  }
  __ec = pthread_mutexattr_destroy(&attr);
  if (__ec) {
    pthread_mutex_destroy(__m);
    return __ec;
  }
  return 0;
}

int __libcpp_recursive_mutex_lock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_mutex_lock(__m);
}

bool __libcpp_recursive_mutex_trylock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return true;
  return pthread_mutex_trylock(__m) == 0;
}

int __libcpp_recursive_mutex_unlock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_mutex_unlock(__m);
}

int __libcpp_recursive_mutex_destroy(__libcpp_recursive_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  return pthread_mutex_destroy(__m);
}

int __libcpp_mutex_lock(__libcpp_mutex_t* __m) {
  // All libcxx-internal locks are released before we run any code which could
  // spawn a thread, so we can safely skip mutex acquisition when there's only
  // one thread (even if the threading API is enabled).
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_mutex_lock(__m);
}

bool __libcpp_mutex_trylock(__libcpp_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return true;
  return pthread_mutex_trylock(__m) == 0;
}

int __libcpp_mutex_unlock(__libcpp_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_mutex_unlock(__m);
}

int __libcpp_mutex_destroy(__libcpp_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  return pthread_mutex_destroy(__m);
}

// Condition Variable
int __libcpp_condvar_signal(__libcpp_condvar_t* __cv) {
  // If we're the only thread, there's no one to signal to, skip it
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_cond_signal(__cv);
}

int __libcpp_condvar_broadcast(__libcpp_condvar_t* __cv) {
  // If we're the only thread, there's no one to broadcast to, skip it
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return pthread_cond_broadcast(__cv);
}

int __libcpp_condvar_wait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m) {
  // If we're the only thread, there's no one to wake us up, so this is a deadlock
  assert(__libcpp_might_have_multiple_threads());
  return pthread_cond_wait(__cv, __m);
}

int __libcpp_condvar_timedwait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m, __libcpp_timespec_t* __ts) {
  if (!__libcpp_is_threading_api_enabled()) {
    // With nobody to wake us up, this is equivalent to a sleep
    // TODO: actually wait until __ts, and replace this with
    // `if(!__libcpp_might_have_multiple_threads())`
    return ETIMEDOUT;
  }
  return pthread_cond_timedwait(__cv, __m, __ts);
}

int __libcpp_condvar_destroy(__libcpp_condvar_t* __cv) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  return pthread_cond_destroy(__cv);
}

// Execute once
int __libcpp_execute_once(__libcpp_exec_once_flag* flag, void (*init_routine)()) {
  if (!__libcpp_is_threading_api_enabled()) {
    if (*flag == _LIBCPP_EXEC_ONCE_INITIALIZER) {
      init_routine();
      // TODO: In order for this to work when __libcpp_is_threading_api_enabled() can change during
      // program execution, we have to write the same value pthread_once would.
      // For glibc this seems to be 2, but it could vary.
      *flag = 2;
    }
    return 0;
  }

  return pthread_once(flag, init_routine);
}

// Thread id
// Returns non-zero if the thread ids are equal, otherwise 0
bool __libcpp_thread_id_equal(__libcpp_thread_id t1, __libcpp_thread_id t2) { return t1 == t2; }

// Returns non-zero if t1 < t2, otherwise 0
bool __libcpp_thread_id_less(__libcpp_thread_id t1, __libcpp_thread_id t2) { return t1 < t2; }

// Thread
bool __libcpp_thread_isnull(const __libcpp_thread_t* __t) { return __libcpp_thread_get_id(__t) == 0; }

int __libcpp_thread_create(__libcpp_thread_t* __t, void* (*__func)(void*), void* __arg) {
  assert(__libcpp_is_threading_api_enabled());
  return pthread_create(__t, nullptr, __func, __arg);
}

__libcpp_thread_id __libcpp_thread_get_current_id() {
  assert(__libcpp_is_threading_api_enabled());
  const __libcpp_thread_t thread = pthread_self();
  return __libcpp_thread_get_id(&thread);
}

__libcpp_thread_id __libcpp_thread_get_id(const __libcpp_thread_t* __t) {
#    if defined(__MVS__)
  return __t->__;
#    else
  return *__t;
#    endif
}

int __libcpp_thread_join(__libcpp_thread_t* __t) {
  assert(__libcpp_is_threading_api_enabled());
  return pthread_join(*__t, nullptr);
}

int __libcpp_thread_detach(__libcpp_thread_t* __t) {
  assert(__libcpp_is_threading_api_enabled());
  return pthread_detach(*__t);
}

void __libcpp_thread_yield() {
  if (!__libcpp_might_have_multiple_threads())
    return;
  sched_yield();
}

void __libcpp_thread_sleep_for(const chrono::nanoseconds& __ns) {
  assert(__libcpp_is_threading_api_enabled());
  __libcpp_timespec_t __ts = __thread_detail::__convert_to_timespec(__ns);
  while (nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
    ;
}

// Thread local storage
int __libcpp_tls_create(__libcpp_tls_key* __key, void (*__at_exit)(void*)) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  return pthread_key_create(__key, __at_exit);
}

void* __libcpp_tls_get(__libcpp_tls_key __key) {
  // ugly and devious way of getting cxa_exception_storage to work, but necessary if we want to
  // keep all the runtime-dependent threading changes contained within internal_threading_support.h
  if (!__libcpp_is_threading_api_enabled()) {
    static struct {
      void* caughtExceptions;
      unsigned int uncaughtExceptions;
    } eh_globals;
    return &eh_globals;
  }
  // TODO: this won't work if __libcpp_is_threading_api_enabled() can change during program
  // execution, because we may have skipped pthread_key_create()
  return pthread_getspecific(__key);
}

int __libcpp_tls_set(__libcpp_tls_key __key, void* __p) {
  assert(__libcpp_is_threading_api_enabled());
  return pthread_setspecific(__key, __p);
}

#  elif defined(_LIBCPP_HAS_THREAD_API_C11)

int __libcpp_recursive_mutex_init(__libcpp_recursive_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;

  return mtx_init(__m, mtx_plain | mtx_recursive) == thrd_success ? 0 : EINVAL;
}

int __libcpp_recursive_mutex_lock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return mtx_lock(__m) == thrd_success ? 0 : EINVAL;
}

bool __libcpp_recursive_mutex_trylock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return true;
  return mtx_trylock(__m) == thrd_success;
}

int __libcpp_recursive_mutex_unlock(__libcpp_recursive_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return mtx_unlock(__m) == thrd_success ? 0 : EINVAL;
}

int __libcpp_recursive_mutex_destroy(__libcpp_recursive_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  mtx_destroy(__m);
  return 0;
}

int __libcpp_mutex_lock(__libcpp_mutex_t* __m) {
  // All libcxx-internal locks are released before we run any code which could
  // spawn a thread, so we can safely skip mutex acquisition when there's only
  // one thread (even if the threading API is enabled).
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return mtx_lock(__m) == thrd_success ? 0 : EINVAL;
}

bool __libcpp_mutex_trylock(__libcpp_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return true;
  return mtx_trylock(__m) == thrd_success;
}

int __libcpp_mutex_unlock(__libcpp_mutex_t* __m) {
  // See __libcpp_mutex_lock for an explaination of why this is safe
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return mtx_unlock(__m) == thrd_success ? 0 : EINVAL;
}

int __libcpp_mutex_destroy(__libcpp_mutex_t* __m) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  mtx_destroy(__m);
  return 0;
}

// Condition Variable
int __libcpp_condvar_signal(__libcpp_condvar_t* __cv) {
  // If we're the only thread, there's no one to signal to, skip it
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return cnd_signal(__cv) == thrd_success ? 0 : EINVAL;
}

int __libcpp_condvar_broadcast(__libcpp_condvar_t* __cv) {
  // If we're the only thread, there's no one to broadcast to, skip it
  if (!__libcpp_might_have_multiple_threads())
    return 0;
  return cnd_broadcast(__cv) == thrd_success ? 0 : EINVAL;
}

int __libcpp_condvar_wait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m) {
  // If we're the only thread, there's no one to wake us up, so this is a deadlock
  assert(__libcpp_might_have_multiple_threads());
  return cnd_wait(__cv, __m) == thrd_success ? 0 : EINVAL;
}

int __libcpp_condvar_timedwait(__libcpp_condvar_t* __cv, __libcpp_mutex_t* __m, timespec* __ts) {
  if (!__libcpp_is_threading_api_enabled()) {
    // With nobody to wake us up, this is equivalent to a sleep
    // TODO: actually wait until __ts, and replace this with
    // `if(!__libcpp_might_have_multiple_threads())`
    return ETIMEDOUT;
  }
  int __ec = cnd_timedwait(__cv, __m, __ts);
  return __ec == thrd_timedout ? ETIMEDOUT : __ec;
}

int __libcpp_condvar_destroy(__libcpp_condvar_t* __cv) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  cnd_destroy(__cv);
  return 0;
}

// Execute once
int __libcpp_execute_once(__libcpp_exec_once_flag* flag, void (*init_routine)(void)) {
  if (!__libcpp_is_threading_api_enabled()) {
    if (*flag == _LIBCPP_EXEC_ONCE_INITIALIZER) {
      init_routine();
      // TODO: In order for this to work when __libcpp_is_threading_api_enabled() can change during
      // program execution, we have to write the same value ::call_once would.
      // For glibc this seems to be 2, but it could vary.
      *flag = 2;
    }
    return 0;
  }

  ::call_once(flag, init_routine);
  return 0;
}

// Thread id
// Returns non-zero if the thread ids are equal, otherwise 0
bool __libcpp_thread_id_equal(__libcpp_thread_id t1, __libcpp_thread_id t2) { return thrd_equal(t1, t2) != 0; }

// Returns non-zero if t1 < t2, otherwise 0
bool __libcpp_thread_id_less(__libcpp_thread_id t1, __libcpp_thread_id t2) { return t1 < t2; }

// Thread
bool __libcpp_thread_isnull(const __libcpp_thread_t* __t) { return __libcpp_thread_get_id(__t) == 0; }

int __libcpp_thread_create(__libcpp_thread_t* __t, void* (*__func)(void*), void* __arg) {
  assert(__libcpp_is_threading_api_enabled());
  int __ec = thrd_create(__t, reinterpret_cast<thrd_start_t>(__func), __arg);
  return __ec == thrd_nomem ? ENOMEM : __ec;
}

__libcpp_thread_id __libcpp_thread_get_current_id() {
  assert(__libcpp_is_threading_api_enabled());
  return thrd_current();
}

__libcpp_thread_id __libcpp_thread_get_id(const __libcpp_thread_t* __t) { return *__t; }

int __libcpp_thread_join(__libcpp_thread_t* __t) {
  assert(__libcpp_is_threading_api_enabled());
  return thrd_join(*__t, nullptr) == thrd_success ? 0 : EINVAL;
}

int __libcpp_thread_detach(__libcpp_thread_t* __t) {
  assert(__libcpp_is_threading_api_enabled());
  return thrd_detach(*__t) == thrd_success ? 0 : EINVAL;
}

void __libcpp_thread_yield() {
  if (!__libcpp_might_have_multiple_threads())
    return;
  thrd_yield();
}

void __libcpp_thread_sleep_for(const chrono::nanoseconds& __ns) {
  assert(__libcpp_is_threading_api_enabled());
  __libcpp_timespec_t __ts = __thread_detail::__convert_to_timespec(__ns);
  thrd_sleep(&__ts, nullptr);
}

// Thread local storage
int __libcpp_tls_create(__libcpp_tls_key* __key, void (*__at_exit)(void*)) {
  if (!__libcpp_is_threading_api_enabled())
    return 0;
  return tss_create(__key, __at_exit) == thrd_success ? 0 : EINVAL;
}

void* __libcpp_tls_get(__libcpp_tls_key __key) {
  // ugly and devious way of getting cxa_exception_storage to work, but necessary if we want to
  // keep all the runtime-dependent threading changes contained within internal_threading_support.h
  if (!__libcpp_is_threading_api_enabled()) {
    static struct {
      void* caughtExceptions;
      unsigned int uncaughtExceptions;
    } eh_globals;
    return &eh_globals;
  }
  // TODO: this won't work if __libcpp_is_threading_api_enabled() can change during program
  // execution, because we may have skipped tss_create()
  return tss_get(__key);
}

int __libcpp_tls_set(__libcpp_tls_key __key, void* __p) {
  assert(__libcpp_is_threading_api_enabled());
  return tss_set(__key, __p) == thrd_success ? 0 : EINVAL;
}

#  endif

#endif // _LIBCPP_HAS_THREAD_LIBRARY_OPTIONAL && !_LIBCPP_HAS_THREAD_LIBRARY_EXTERNAL

#endif // _LIBCPP_INTERNAL_THREADING_SUPPORT_H
