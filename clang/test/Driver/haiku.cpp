// Check the C++ header path when using libc++
// RUN: %clang++ --target=x86_64-unknown-haiku --stdlib=libc++ -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LIBCXX-HEADER-PATH %s
// CHECK-LIBCXX-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/c++/v1"
