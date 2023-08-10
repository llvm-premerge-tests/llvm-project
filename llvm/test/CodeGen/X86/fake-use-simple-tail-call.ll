; RUN: llc < %s -mtriple=x86_64-unknown-unknown -O2 -o - \
; RUN:   | FileCheck %s --implicit-check-not=TAILCALL
; Generated with: clang -emit-llvm -O2 -S -fextend-lifetimes test.cpp -o -
; =========== test.cpp ===============
; extern int bar(int);
; int foo1(int i)
; {
;     return bar(i);
; }
; =========== test.cpp ===============

; CHECK: TAILCALL

; ModuleID = 'test.cpp'
source_filename = "test.cpp"

define i32 @_Z4foo1i(i32 %i) local_unnamed_addr {
entry:
  %call = tail call i32 @_Z3bari(i32 %i)
  tail call void (...) @llvm.fake.use(i32 %i)
  ret i32 %call
}

declare i32 @_Z3bari(i32) local_unnamed_addr

declare void @llvm.fake.use(...)

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 5.0.1"}
