; RUN: opt -mtriple=x86_64-unknown-unknown -S -codegenprepare <%s -o - | FileCheck %s
;
; Ensure return instruction splitting ignores fake uses.
;
; IR Generated with clang -O2 -S -emit-llvm -fextend-lifetimes test.cpp
;
;// test.cpp
;extern int bar(int);
;
;int foo2(int i)
;{
;  --i;
;  if (i <= 0)
;    return -1;
;  return bar(i);
;}

; ModuleID = 'test.cpp'
source_filename = "test.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

declare i32 @_Z3bari(i32) local_unnamed_addr

; Function Attrs: nounwind
declare void @llvm.fake.use(...)

; Function Attrs: nounwind sspstrong uwtable
define i32 @_Z4foo2i(i32 %i) local_unnamed_addr {
entry:
  %dec = add nsw i32 %i, -1
  %cmp = icmp slt i32 %i, 2
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @_Z3bari(i32 %dec)
; CHECK: ret i32 %call
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
; CHECK: cleanup:
  %retval.0 = phi i32 [ %call, %if.end ], [ -1, %entry ]
  tail call void (...) @llvm.fake.use(i32 %dec)
; CHECK: ret i32 -1
  ret i32 %retval.0
}

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 2}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 7.0.0"}
