; RUN: opt -O2 -S %s | FileCheck %s
;
; Check that we do not assert and that we retain the fake_use
; instruction that uses the address of bar.
;
; CHECK: define{{.*}}foo
; CHECK: call{{.*llvm\.fake\.use.*}}%bar
;
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

define void @_Z3fooPi(ptr %bar) {
entry:
  %bar.addr = alloca ptr, align 8
  %baz = alloca ptr, align 8
  store ptr %bar, ptr %bar.addr, align 8
  %0 = bitcast ptr %bar.addr to ptr
  store ptr %bar.addr, ptr %baz, align 8
  %1 = load ptr, ptr %bar.addr, align 8
  call void (...) @llvm.fake.use(ptr %1)
  %2 = load ptr, ptr %baz, align 8
  call void (...) @llvm.fake.use(ptr %2)
  ret void
}

declare void @llvm.fake.use(...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.9.0"}
