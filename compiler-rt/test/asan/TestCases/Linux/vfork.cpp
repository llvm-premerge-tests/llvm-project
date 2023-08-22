// https://github.com/google/sanitizers/issues/925
// RUN: %clang_asan -O0 %s -o %t && %run %t 2>&1

// REQUIRES: target={{(aarch64|arm|i386|loongarch64|mips.*|riscv64|x86_64)-.*}}

#include <assert.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <sanitizer/asan_interface.h>

__attribute__((noinline, no_sanitize("address"))) void child() {
  alignas(8) char x[100000];
  __asan_poison_memory_region(x, sizeof(x));
  _exit(0);
}

__attribute__((noinline, no_sanitize("address"))) void parent() {
  alignas(8) char x[100000];
  assert(__asan_address_is_poisoned(x + 5000) == 0);
}

int main(int argc, char **argv) {
  if (vfork())
    parent();
  else
    child();

  return 0;
}
