#ifndef __LLVM_LIBC_MACROS_LIMITS_MACROS_H
#define __LLVM_LIBC_MACROS_LIMITS_MACROS_H

#ifdef __linux__
#include <linux/limits.h>
#endif

#if __LP64__
#define __LONG_MAX 0x7fffffffffffffffL
#else
#define __LONG_MAX 0x7fffffffL
#endif // __LP64__

#define LONG_MAX __LONG_MAX
#define SSIZE_MAX LONG_MAX

#endif // __LLVM_LIBC_MACROS_LIMITS_MACROS_H
