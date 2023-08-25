;; Check the cases involving internal CFI instrumented functions where we do not expect functions to be merged.
; RUN: opt -S -passes=mergefunc < %s | FileCheck %s


target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-linux-android28"

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i1 @llvm.type.test(ptr, metadata) #6

define internal void @A__on_zero_sharedEv(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #3 align 2 {
; CHECK-LABEL: @A__on_zero_sharedEv
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %vtable = load ptr, ptr %this1, align 8
  %0 = call i1 @llvm.type.test(ptr %vtable, metadata !11), !nosanitize !47
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define internal void @B__on_zero_sharedEv(ptr noundef nonnull align 8 dereferenceable(32) %this) unnamed_addr #3 align 2 {
; CHECK-LABEL: @B__on_zero_sharedEv
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %vtable = load ptr, ptr %this1, align 8
  %0 = call i1 @llvm.type.test(ptr %vtable, metadata !22), !nosanitize !47
  ret void
}

attributes #3 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fix-cortex-a53-835769,+neon,+outline-atomics,+v8a" }
attributes #6 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!10 = !{i64 16, !11}
!11 = distinct !{}
!21 = !{i64 16, !22}
!22 = distinct !{}
!47 = !{}
