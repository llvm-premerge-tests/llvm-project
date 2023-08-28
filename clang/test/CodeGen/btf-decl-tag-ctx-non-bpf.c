// REQUIRES: x86-registered-target
// RUN: %clang -cc1 -triple x86_64 -disable-llvm-passes -S -emit-llvm  -o - %s \
// RUN: | FileCheck %s

// Verify that btf_decl_tag("ctx") has no effect for non-BPF target.

#define __ctx __attribute__((btf_decl_tag("ctx")))

struct foo {
  int a;
} __ctx;

// CHECK-NOT: @llvm.context.marker.bpf

int bar(struct foo *p) {
  return p->a;
}
