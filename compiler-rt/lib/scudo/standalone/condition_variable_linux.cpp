//===-- condition_variable_linux.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_LINUX

#include "condition_variable_linux.h"

#include "atomic_helpers.h"

#include <limits.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace scudo {

void ConditionVariableLinux::notifyAllImpl(UNUSED HybridMutex &M) {
  u32 V = atomic_fetch_add(&Counter, 1U, memory_order_acquire);

  // TODO(chiahungduan): Move the waiters from the futex waiting queue
  // `Counter` to futex waiting queue `M` so that the awoken threads won't be
  // blocked again due to locked `M` by current thread.
  if (V != LastNotifyAll) {
    syscall(SYS_futex, reinterpret_cast<uptr>(&Counter), FUTEX_WAKE_PRIVATE,
            INT_MAX, nullptr, nullptr, 0);
  }

  LastNotifyAll = V + 1;
}

void ConditionVariableLinux::waitImpl(HybridMutex &M) {
  u32 V = atomic_fetch_add(&Counter, 1U, memory_order_acq_rel);

  // TODO: Use ScopedUnlock when it's supported.
  M.unlock();
  syscall(SYS_futex, reinterpret_cast<uptr>(&Counter), FUTEX_WAIT_PRIVATE,
          V + 1, nullptr, nullptr, 0);
  M.lock();
}

} // namespace scudo

#endif // SCUDO_LINUX
