; RUN: opt -S -passes=hotcoldsplit < %s | FileCheck %s

; Check that the unlikely branch is outlined.
; int cold(const char*);
; int hot(const char*);
; void foo(int a, int b) {
;   if (a == b) [[unlikely]] { // Should be outlined.
;     cold("same");
;     cold("same");
;   } else {
;     hot("different");
;   }
; }

; void bar(int a, int b) {
;   if (a == b) [[likely]] {
;     hot("same");
;   } else { // Should be outlined.
;     cold("different");
;     cold("different");
;   }
; }


; CHECK: internal void @foo.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-NEXT: newFuncRoot
; CHECK: tail call noundef i32 @cold
; CHECK: tail call noundef i32 @cold

; CHECK: internal void @bar.cold.1() #[[ATTR0:[0-9]+]]
; CHECK-NEXT: newFuncRoot
; CHECK: tail call noundef i32 @cold
; CHECK: tail call noundef i32 @cold
; attributes #[[ATTR0]] = { cold minsize }

source_filename = "/app/example.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"same\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"different\00", align 1

define dso_local void @foo(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !1

if.then:
  %call = tail call noundef i32 @cold(ptr noundef nonnull @.str)
  %call1 = tail call noundef i32 @cold(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call2 = tail call noundef i32 @hot(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

declare noundef i32 @cold(ptr noundef) local_unnamed_addr #1

declare noundef i32 @hot(ptr noundef) local_unnamed_addr #1

define dso_local void @bar(i32 noundef %a, i32 noundef %b) local_unnamed_addr {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else, !prof !2

if.then:
  %call = tail call noundef i32 @hot(ptr noundef nonnull @.str)
  br label %if.end

if.else:
  %call1 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  %call2 = tail call noundef i32 @cold(ptr noundef nonnull @.str.1)
  br label %if.end

if.end:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}