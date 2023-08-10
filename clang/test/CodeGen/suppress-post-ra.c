// RUN: %clang_cc1 %s -S -emit-llvm -O2 -fextend-lifetimes -o - | FileCheck %s

// Emit the function attribute disable-post-ra when
// -fextend-lifetimes is on.

// CHECK: attributes #0 = {{{.*}}"disable-post-ra"

void foo() {}
