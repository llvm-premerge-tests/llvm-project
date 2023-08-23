// RUN: %clang_cc1 -emit-llvm -triple x86_64 -std=c17 -fsanitize=function %s -o - | FileCheck %s
// RUN: not %clang_cc1 -emit-llvm -triple x86_64-sie-ps5 -fsanitize=function %s -o 2>&1 | FileCheck %s --check-prefix=UBSAN-FUNCTION-ERR
// RUN: not %clang_cc1 -emit-llvm -triple armv6t2-unknown-unknown-eabi -target-feature +execute-only -fsanitize=function %s -o 2>&1 | FileCheck %s --check-prefix=UBSAN-FUNCTION-ERR

// CHECK-LABEL: define{{.*}} @call_no_prototype(
// CHECK-NOT:     __ubsan_handle_function_type_mismatch
void call_no_prototype(void (*f)()) { f(); }

// CHECK-LABEL: define{{.*}} @call_prototype(
// CHECK:         __ubsan_handle_function_type_mismatch
void call_prototype(void (*f)(void)) { f(); }

// UBSAN-FUNCTION-ERR: error: unsupported option '-fsanitize=function' for the execute only target {{('x86_64-sie-ps5'|'armv6t2-unknown-unknown-eabi')}}
