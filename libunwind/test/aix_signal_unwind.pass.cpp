// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _Unwind_Backtrace() walks from a signal handler and produces
// a correct traceback when the function raising the signal is a leaf function
// in the middle of the call chain.

// REQUIRES: target=powerpc{{(64)?}}-ibm-aix

#undef NDEBUG
#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/debug.h>
#include <unwind.h>

#define FUNC_ARRAY_SIZE 10

struct func_t {
  const char *name1; // Function name obtained in the function.
  char *name2;       // Function name obtained during unwind.
} func[FUNC_ARRAY_SIZE];

int funcIndex = 0;
int lastFuncIndex = 0;

// Get the function name from traceback table.
char *getFuncName(uintptr_t pc, uint16_t *nameLen) {
  uint32_t *p = reinterpret_cast<uint32_t *>(pc);

  // Keep looking forward until a word of 0 is found. The traceback
  // table starts at the following word.
  while (*p)
    ++p;
  tbtable *TBTable = reinterpret_cast<tbtable *>(p + 1);

  if (!TBTable->tb.name_present)
    return NULL;

  // Get to the optional portion of the traceback table.
  p = reinterpret_cast<uint32_t *>(&TBTable->tb_ext);

  // Skip field parminfo if it exists.
  if (TBTable->tb.fixedparms || TBTable->tb.floatparms)
    ++p;

  // Skip field tb_offset if it exists.
  if (TBTable->tb.has_tboff)
    ++p;

  // Skip field hand_mask if it exists.
  if (TBTable->tb.int_hndl)
    ++p;

  // Skip fields ctl_info and ctl_info_disp if they exist.
  if (TBTable->tb.has_ctl)
    p += 1 + *p;

  *nameLen = *reinterpret_cast<uint16_t *>(p);
  return reinterpret_cast<char *>(p) + sizeof(uint16_t);
}

_Unwind_Reason_Code callBack(struct _Unwind_Context *uc, void *arg) {
  (void)arg;
  if (funcIndex >= 0) {
    uint16_t nameLen;
    uintptr_t ip = _Unwind_GetIP(uc);
    func[funcIndex--].name2 = strndup(getFuncName(ip, &nameLen), nameLen);
  }
  return _URC_NO_REASON;
}

extern "C" void handler(int signum) {
  (void)signum;
  func[funcIndex].name1 = __func__;
  lastFuncIndex = funcIndex;

  // Walk stack frames for traceback.
  _Unwind_Backtrace(callBack, NULL);

  // Verify the traceback.
  for (int i = 0; i <= lastFuncIndex; ++i) {
    assert(!strcmp(func[i].name1, func[i].name2) &&
           "Function names do not match");
    free(func[i].name2);
  }
  exit(0);
}

volatile int *null = 0;

// abc() is a leaf function that raises signal SIGSEGV.
extern "C" __attribute__((noinline)) void abc() {
  func[funcIndex++].name1 = __func__;
  // Produce a SIGSEGV.
  *null = 0;
}

extern "C" __attribute__((noinline)) void bar() {
  func[funcIndex++].name1 = __func__;
  abc();
}

extern "C" __attribute__((noinline)) void foo() {
  func[funcIndex++].name1 = __func__;
  bar();
}

int main() {
  // Set signal handler for SIGSEGV.
  signal(SIGSEGV, handler);

  func[funcIndex++].name1 = __func__;
  foo();
}
