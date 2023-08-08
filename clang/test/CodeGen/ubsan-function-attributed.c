// RUN: %clang_cc1 -emit-llvm -triple x86_64 -std=c17 -fsanitize=function %s -o /dev/null

long __attribute__((ms_abi)) f() {}
