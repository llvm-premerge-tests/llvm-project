; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -S -o %t 2>&1 | FileCheck %s --check-prefix=ALIAS
; RUN: FileCheck %s <%t
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "arm-apple-ios"

; CHECK: Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
; CHECK-NEXT: declare void @llvm.memset.p0.i64
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #0
; CHECK: Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
; CHECK-NEXT: declare void @llvm.memcpy.p0.p0.i64
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #0
; CHECK: Function Attrs: nosync nounwind willreturn memory(argmem: readwrite)
; CHECK-NEXT: declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) #0

; CHECK: Function Attrs: noinline nounwind memory(read)
; CHECK-NEXT: declare void @a_readonly_func(ptr)
declare void @a_readonly_func(ptr) #1

; CHECK: Function Attrs: noinline nounwind memory(write)
; CHECK-NEXT: declare void @a_writeonly_func(ptr)
declare void @a_writeonly_func(ptr) #2

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2(ptr %P, ptr %Q)
define void @test2(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2:

; ALIAS:   MayAlias:     i8* %P, i8* %Q
; ALIAS:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS:   Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS:   Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2_atomic(ptr %P, ptr %Q)
define void @test2_atomic(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
  ret void

; ALIAS-LABEL: Function: test2_atomic:

; ALIAS:   MayAlias:     i8* %P, i8* %Q
; ALIAS:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; ALIAS:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; ALIAS:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; ALIAS:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; ALIAS:   Both ModRef:   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1) <->   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; ALIAS:   Both ModRef:   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1) <->   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2a(ptr noalias %P, ptr noalias %Q)
define void @test2a(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2a:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2b(ptr noalias %P, ptr noalias %Q)
define void @test2b(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 12
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2b:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: NoAlias:      i8* %P, i8* %R
; ALIAS: NoAlias:      i8* %Q, i8* %R
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2c(ptr noalias %P, ptr noalias %Q)
define void @test2c(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 11
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2c:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: NoAlias:      i8* %P, i8* %R
; ALIAS: NoAlias:      i8* %Q, i8* %R
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2d(ptr noalias %P, ptr noalias %Q)
define void @test2d(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 -12
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2d:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: NoAlias:      i8* %P, i8* %R
; ALIAS: NoAlias:      i8* %Q, i8* %R
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test2e(ptr noalias %P, ptr noalias %Q)
define void @test2e(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 -11
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test2e:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: NoAlias:      i8* %P, i8* %R
; ALIAS: NoAlias:      i8* %Q, i8* %R
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test3(ptr %P, ptr %Q)
define void @test3(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test3:

; ALIAS: MayAlias:     i8* %P, i8* %Q
; ALIAS: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test3a(ptr noalias %P, ptr noalias %Q)
define void @test3a(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test3a:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test4(ptr %P, ptr noalias %Q)
define void @test4(ptr %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test4:

; ALIAS: NoAlias:      i8* %P, i8* %Q
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %Q        <->  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test5(ptr %P, ptr %Q, ptr %R)
define void @test5(ptr %P, ptr %Q, ptr %R) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test5:

; ALIAS: MayAlias:     i8* %P, i8* %Q
; ALIAS: MayAlias:     i8* %P, i8* %R
; ALIAS: MayAlias:     i8* %Q, i8* %R
; ALIAS: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Both ModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test5a(ptr noalias %P, ptr noalias %Q, ptr noalias %R)
define void @test5a(ptr noalias %P, ptr noalias %Q, ptr noalias %R) nounwind ssp {
  load i8, ptr %P
  load i8, ptr %Q
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
  ret void

; ALIAS-LABEL: Function: test5a:

; ALIAS: NoAlias:     i8* %P, i8* %Q
; ALIAS: NoAlias:     i8* %P, i8* %R
; ALIAS: NoAlias:     i8* %Q, i8* %R
; ALIAS: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; ALIAS: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: NoModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; ALIAS: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test6(ptr %P)
define void @test6(ptr %P) #3 {
  load i8, ptr %P
  call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
  call void @a_readonly_func(ptr %P)
  ret void

; ALIAS-LABEL: Function: test6:

; ALIAS: Just Mod:  Ptr: i8* %P        <->  call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
; ALIAS: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(ptr %P)
; ALIAS: Just Mod:   call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false) <->   call void @a_readonly_func(ptr %P)
; ALIAS: Just Ref:   call void @a_readonly_func(ptr %P) <->   call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
}

; CHECK: Function Attrs: nounwind ssp
; CHECK-NEXT: define void @test7(ptr %P)
define void @test7(ptr %P) #3 {
  load i8, ptr %P
  call void @a_writeonly_func(ptr %P)
  call void @a_readonly_func(ptr %P)
  ret void

; ALIAS-LABEL: Function: test7:

; ALIAS: Just Mod:  Ptr: i8* %P        <->  call void @a_writeonly_func(ptr %P)
; ALIAS: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(ptr %P)
; ALIAS: Just Mod:   call void @a_writeonly_func(ptr %P) <->   call void @a_readonly_func(ptr %P)
; ALIAS: Just Ref:   call void @a_readonly_func(ptr %P) <->   call void @a_writeonly_func(ptr %P)
}

; CHECK: Function Attrs: nounwind memory(inaccessiblemem: readwrite)
; CHECK-NEXT: declare void @an_inaccessiblememonly_func()
declare void @an_inaccessiblememonly_func() #4
; CHECK: Function Attrs: nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
; CHECK-NEXT: declare void @an_inaccessibleorargmemonly_func(ptr)
declare void @an_inaccessibleorargmemonly_func(ptr) #5
; CHECK: Function Attrs: nounwind memory(argmem: readwrite)
; CHECK-NEXT: declare void @an_argmemonly_func(ptr)
declare void @an_argmemonly_func(ptr) #0

; CHECK-NOT: Function Attrs: 
; CHECK: define void @test8(ptr %p)
define void @test8(ptr %p) {
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p)
  call void @an_inaccessiblememonly_func()
  call void @a_writeonly_func(ptr %q)
  call void @an_inaccessiblememonly_func()
  call void @an_inaccessibleorargmemonly_func(ptr %q)
  call void @an_argmemonly_func(ptr %q)
  ret void

; ALIAS-LABEL: Function: test8
; ALIAS: NoModRef:  Ptr: i8* %p <->  call void @an_inaccessiblememonly_func()
; ALIAS: NoModRef:  Ptr: i8* %q <->  call void @an_inaccessiblememonly_func()
; ALIAS: Both ModRef:  Ptr: i8* %p <->  call void @an_inaccessibleorargmemonly_func(ptr %q)
; ALIAS: Both ModRef:  Ptr: i8* %q <->  call void @an_inaccessibleorargmemonly_func(ptr %q)
; ALIAS: Both ModRef:  Ptr: i8* %p <->  call void @an_argmemonly_func(ptr %q)
; ALIAS: Both ModRef:  Ptr: i8* %q <->  call void @an_argmemonly_func(ptr %q)
; ALIAS: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_inaccessiblememonly_func()
; ALIAS: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; ALIAS: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_argmemonly_func(ptr %q)
; ALIAS: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_readonly_func(ptr %p)
; ALIAS: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_writeonly_func(ptr %q)
; ALIAS: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessiblememonly_func()
; ALIAS: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; ALIAS: NoModRef: call void @an_inaccessiblememonly_func() <-> call void @an_argmemonly_func(ptr %q)
; ALIAS: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; ALIAS: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; ALIAS: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_argmemonly_func(ptr %q)
; ALIAS: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @a_readonly_func(ptr %p)
; ALIAS: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @a_writeonly_func(ptr %q)
; ALIAS: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; ALIAS: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @an_argmemonly_func(ptr %q)
; ALIAS: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @a_readonly_func(ptr %p)
; ALIAS: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @a_writeonly_func(ptr %q)
; ALIAS: NoModRef: call void @an_argmemonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; ALIAS: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
}

;; test that MustAlias is set for calls when no MayAlias is found.
; CHECK: Function Attrs: nounwind memory(argmem: readwrite)
; CHECK-NEXT: declare void @another_argmemonly_func(ptr, ptr)
declare void @another_argmemonly_func(ptr, ptr) #0

; CHECK-NOT: Function Attrs: 
; CHECK: define void @test8a(ptr noalias %p, ptr noalias %q)
define void @test8a(ptr noalias %p, ptr noalias %q) {
entry:
  load i8, ptr %p
  load i8, ptr %q
  call void @another_argmemonly_func(ptr %p, ptr %q)
  ret void

; ALIAS-LABEL: Function: test8a
; ALIAS: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(ptr %p, ptr %q)
; ALIAS: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(ptr %p, ptr %q)
}

; CHECK-NOT: Function Attrs: 
; CHECK: define void @test8b(ptr %p, ptr %q)
define void @test8b(ptr %p, ptr %q) {
entry:
  load i8, ptr %p
  load i8, ptr %q
  call void @another_argmemonly_func(ptr %p, ptr %q)
  ret void

; ALIAS-LABEL: Function: test8b
; ALIAS: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(ptr %p, ptr %q)
; ALIAS: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(ptr %p, ptr %q)
}


;; test that unknown operand bundle has unknown effect to the heap
; CHECK-NOT: Function Attrs: 
; CHECK: define void @test9(ptr %p)
define void @test9(ptr %p) {
; ALIAS-LABEL: Function: test9
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p) [ "unknown"() ]
  call void @an_inaccessiblememonly_func() [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
  ret void

; ALIAS: Both ModRef:  Ptr: i8* %p     <->  call void @a_readonly_func(ptr %p) [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @a_readonly_func(ptr %p) [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %p     <->  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; ALIAS: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
}

;; test callsite overwrite of unknown operand bundle
; CHECK-NOT: Function Attrs: 
; CHECK: define void @test10(ptr %p)
define void @test10(ptr %p) {
; ALIAS-LABEL: Function: test10
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p) #6 [ "unknown"() ]
  call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(ptr %q) #8 [ "unknown"() ]
  call void @an_argmemonly_func(ptr %q) #9 [ "unknown"() ]
  ret void

; ALIAS: Just Ref:  Ptr: i8* %p        <->  call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; ALIAS: Just Ref:  Ptr: i8* %q        <->  call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; ALIAS: NoModRef:  Ptr: i8* %p        <->  call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; ALIAS: NoModRef:  Ptr: i8* %q        <->  call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %p        <->  call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %p        <->  call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; ALIAS: Both ModRef:  Ptr: i8* %q     <->  call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; ALIAS: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; ALIAS: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; ALIAS: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; ALIAS: NoModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; ALIAS: NoModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; ALIAS: Both ModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { noinline nounwind readonly }
attributes #2 = { noinline nounwind writeonly }
attributes #3 = { nounwind ssp }
attributes #4 = { inaccessiblememonly nounwind }
attributes #5 = { inaccessiblemem_or_argmemonly nounwind }
attributes #6 = { readonly }
attributes #7 = { inaccessiblememonly }
attributes #8 = { inaccessiblemem_or_argmemonly }
attributes #9 = { argmemonly }
