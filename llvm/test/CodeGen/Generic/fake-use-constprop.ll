; RUN: opt -passes=gvn %s | FileCheck %s
;
; The Global Value Numbering pass (GVN) propagates boolean values
; that are constant in dominated basic blocks to all the uses
; in these basic blocks. However, we don't want the constant propagated
; into fake.use intrinsics since this would render the intrinsic useless
; with respect to keeping the variable live up until the fake.use.
; This test checks that we don't generate any fake.uses with constant 0.
;
; Generate with clang -O2 -S -emit-llvm -fextend-lifetimes test.c
;
; extern void func1();
; extern int bar();
; extern void baz(int);
;
; int foo(int i, float f, int *punused)
; {
;   int j = 3*i;
;   if (j > 0) {
;     int m = bar(i);
;     if (m) {
;       char b = f;
;       baz(b);
;       if (b)
;         goto lab;
;       func1();
;     }
; lab:
;     func1();
;   }
;   return 1;
; }
;
; ModuleID = 'test.c'
;
; CHECK-NOT: fake.use({{.*}} 0)
;
; Function Attrs: nounwind sspstrong uwtable
define i32 @foo(i32 %i, float %f, i32* %punused) {
entry:
  %mul = mul nsw i32 %i, 3
  %cmp = icmp sgt i32 %i, 0
  br i1 %cmp, label %if.then, label %if.end6

if.then:                                          ; preds = %entry
  %call = tail call i32 (i32, ...) bitcast (i32 (...)* @bar to i32 (i32, ...)*)(i32 %i)
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %lab, label %if.then1

if.then1:                                         ; preds = %if.then
  %conv = fptosi float %f to i8
  %conv2 = sext i8 %conv to i32
  tail call void @baz(i32 %conv2)
  %tobool3 = icmp eq i8 %conv, 0
  br i1 %tobool3, label %if.end, label %lab

if.end:                                           ; preds = %if.then1
  tail call void (...) @func1()
  tail call void (...) @llvm.fake.use(i8 signext %conv)
  br label %lab

lab:                                              ; preds = %if.end, %if.then1, %if.then
  tail call void (...) @func1()
  tail call void (...) @llvm.fake.use(i32 %call)
  br label %if.end6

if.end6:                                          ; preds = %lab, %entry
  tail call void (...) @llvm.fake.use(i32 %i)
  tail call void (...) @llvm.fake.use(float %f)
  tail call void (...) @llvm.fake.use(i32* %punused)
  tail call void (...) @llvm.fake.use(i32 %mul)
  ret i32 1
}

declare i32 @bar(...)

declare void @baz(i32)

declare void @func1(...)

; Function Attrs: nounwind
declare void @llvm.fake.use(...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.9.0"}
