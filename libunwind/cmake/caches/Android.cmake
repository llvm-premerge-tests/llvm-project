set(LLVM_ENABLE_RUNTIMES "libunwind" CACHE STRING "")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

# Clang defaults to linking libunwind and libc++ libraries, but when we're
# building libunwind.a, neither of these dependencies exist yet. libunwind's
# CMake files pass --unwindlib=none and -nostdlib++ to avoid this problem, but
# CMake's default "compiler works" tests don't pass those flags, so force those
# tests to pass.
set(CMAKE_C_COMPILER_WORKS ON CACHE BOOL "")
set(CMAKE_CXX_COMPILER_WORKS ON CACHE BOOL "")

# On Android, the NDK's libunwind.a is linked into every NDK binary, and
# libunwind.a hides the symbols so that different unwinders don't interact.
# Starting with Android R, libc.so exports the EH unwinding APIs using a copy of
# the LLVM libunwind.
set(LIBUNWIND_HIDE_SYMBOLS ON CACHE BOOL "")
set(LIBUNWIND_ENABLE_SHARED OFF CACHE BOOL "")
