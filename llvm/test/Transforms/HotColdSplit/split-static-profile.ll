; RUN: opt -S -passes=hotcoldsplit < %s 2>&1  | FileCheck %s

;Check that the unlikely branch is outlined.
;int cold(const char*);
;int hot(const char*);

;void foo(int a, int b) {
;  if (a == b) [[unlikely]] {
;    cold("same");
;    cold("same");
;  } else {
;    hot("different");
;  }
;}

; CHECK: internal void @_Z3fooii.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-NEXT: newFuncRoot
; CHECK: tail call noundef i32 @_Z4coldPKc
; CHECK: tail call noundef i32 @_Z4coldPKc
; attributes #[[ATTR0]] = { cold minsize }

source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"same\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"different\00", align 1

define dso_local void @_Z3fooii(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !6

if.then:
  %call = tail call noundef i32 @_Z4coldPKc(ptr noundef nonnull @.str)
  %call1 = tail call noundef i32 @_Z4coldPKc(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call2 = tail call noundef i32 @_Z3hotPKc(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

declare noundef i32 @_Z4coldPKc(ptr noundef) local_unnamed_addr #1

declare noundef i32 @_Z3hotPKc(ptr noundef) local_unnamed_addr #1

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git 733edde00582c2a4a6266a4cd2bed7015f4246ce)"}
!6 = !{!"branch_weights", i32 1, i32 2000}
