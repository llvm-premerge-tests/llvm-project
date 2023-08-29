; RUN: opt -S -mtriple=aarch64-linux-gnu -opaque-pointers -aarch64-sme-abi %s | FileCheck %s
; RUN: opt -S -mtriple=aarch64-linux-gnu -opaque-pointers -aarch64-sme-abi -aarch64-sme-abi %s | FileCheck %s
; Check that the test also passes when not in opaque-pointer mode.
; RUN: opt -S -mtriple=aarch64-linux-gnu -aarch64-sme-abi %s -o /dev/null

; XFAIL: *

declare void @private_za_callee();
declare void @shared_za_callee() "aarch64_pstate_za_shared"
declare void @new_za_callee() "aarch64_pstate_za_new"
declare void @preserved_za_callee() "aarch64_pstate_za_preserved"
declare i64 @foo(i64);

declare float @llvm.cos.f32(float)

; CHECK: type { ptr, i16, [6 x i8] }

; Shared ZA Caller, Private ZA Callee

define void @shared_za_caller() "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_caller() #4 {
; CHECK-NEXT:    %tpidr2.call.obj = alloca %tpidr2_ty, align 8
; CHECK-NEXT:    %N = call i64 @llvm.aarch64.sme.cntsb()
; CHECK-NEXT:    %NN = mul i64 %N, %N
; CHECK-NEXT:    %buffer = alloca i8, i64 %NN, align 16
; CHECK-NEXT:    %tpidr2.call.obj.buffer = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 0
; CHECK-NEXT:    store ptr %buffer, ptr %tpidr2.call.obj.buffer, align 8
; CHECK-NEXT:    %live = call i64 @llvm.aarch64.sme.get.live.za.slices()
; CHECK-NEXT:    %live.trunc = trunc i64 %live to i16
; CHECK-NEXT:    %tpidr2.obj.live = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 1
; CHECK-NEXT:    store i16 %live.trunc, ptr %tpidr2.obj.live, align 2
; CHECK-NEXT:    %tpi.int = ptrtoint ptr %tpidr2.call.obj to i64
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 %tpi.int)
; CHECK-NEXT:    call void @private_za_callee()
; CHECK-NEXT:    call void @llvm.aarch64.sme.start.pstateza()
; CHECK-NEXT:    %tpidr2 = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    %cmp = icmp eq i64 %tpidr2, 0
; CHECK-NEXT:    br i1 %cmp, label %restore.za, label %resume
; CHECK:       restore.za:
; CHECK-NEXT:    call void @llvm.aarch64.sme.tpidr2.restore(ptr %tpidr2.call.obj)
; CHECK-NEXT:    br label %resume
; CHECK:       resume:
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    ret void
;
  call void @private_za_callee()
  ret void
}

; Shared ZA Caller, Private ZA Callees

define i64 @shared_za_caller_multiple_callees(i64 %a) "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_caller_multiple_callees(i64 %a) #4 {
; CHECK-NEXT:    %tpidr2.call.obj = alloca %tpidr2_ty, align 8
; CHECK-NEXT:    %N = call i64 @llvm.aarch64.sme.cntsb()
; CHECK-NEXT:    %NN = mul i64 %N, %N
; CHECK-NEXT:    %buffer = alloca i8, i64 %NN, align 16
; CHECK-NEXT:    %tpidr2.call.obj.buffer = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 0
; CHECK-NEXT:    store ptr %buffer, ptr %tpidr2.call.obj.buffer, align 8
; CHECK-NEXT:    %live = call i64 @llvm.aarch64.sme.get.live.za.slices()
; CHECK-NEXT:    %live.trunc = trunc i64 %live to i16
; CHECK-NEXT:    %tpidr2.obj.live = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 1
; CHECK-NEXT:    store i16 %live.trunc, ptr %tpidr2.obj.live, align 2
; CHECK-NEXT:    %tpi.int = ptrtoint ptr %tpidr2.call.obj to i64
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 %tpi.int)
; CHECK-NEXT:    %b = call i64 @foo(i64 %a)
; CHECK-NEXT:    call void @llvm.aarch64.sme.start.pstateza()
; CHECK-NEXT:    %tpidr2 = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    %cmp = icmp eq i64 %tpidr2, 0
; CHECK-NEXT:    br i1 %cmp, label %restore.za, label %resume
; CHECK:       restore.za:
; CHECK-NEXT:    call void @llvm.aarch64.sme.tpidr2.restore(ptr %tpidr2.call.obj)
; CHECK-NEXT:    br label %resume
; CHECK:       resume:
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    %sum = add i64 %a, %b
; CHECK-NEXT:    %live3 = call i64 @llvm.aarch64.sme.get.live.za.slices()
; CHECK-NEXT:    %live.trunc4 = trunc i64 %live3 to i16
; CHECK-NEXT:    %tpidr2.obj.live5 = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 1
; CHECK-NEXT:    store i16 %live.trunc4, ptr %tpidr2.obj.live5, align 2
; CHECK-NEXT:    %tpi.int6 = ptrtoint ptr %tpidr2.call.obj to i64
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 %tpi.int6)
; CHECK-NEXT:    %c = call i64 @foo(i64 %sum)
; CHECK-NEXT:    call void @llvm.aarch64.sme.start.pstateza()
; CHECK-NEXT:    %tpidr27 = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    %cmp8 = icmp eq i64 %tpidr27, 0
; CHECK-NEXT:    br i1 %cmp8, label %restore.za2, label %resume1
; CHECK:       restore.za2:
; CHECK-NEXT:    call void @llvm.aarch64.sme.tpidr2.restore(ptr %tpidr2.call.obj)
; CHECK-NEXT:    br label %resume1
; CHECK:       resume1:
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    %res = mul i64 %sum, %c
; CHECK-NEXT:    ret i64 %res
;
  %b = call i64 @foo(i64 %a)
  %sum = add i64 %a, %b
  %c = call i64 @foo(i64 %sum)
  %res = mul i64 %sum, %c
  ret i64 %res
}

; Shared ZA Caller, New ZA Callee

define void @shared_za_new_za_callee() "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_new_za_callee() #4 {
; CHECK-NEXT:    %tpidr2.call.obj = alloca %tpidr2_ty, align 8
; CHECK-NEXT:    %N = call i64 @llvm.aarch64.sme.cntsb()
; CHECK-NEXT:    %NN = mul i64 %N, %N
; CHECK-NEXT:    %buffer = alloca i8, i64 %NN, align 16
; CHECK-NEXT:    %tpidr2.call.obj.buffer = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 0
; CHECK-NEXT:    store ptr %buffer, ptr %tpidr2.call.obj.buffer, align 8
; CHECK-NEXT:    %live = call i64 @llvm.aarch64.sme.get.live.za.slices()
; CHECK-NEXT:    %live.trunc = trunc i64 %live to i16
; CHECK-NEXT:    %tpidr2.obj.live = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 1
; CHECK-NEXT:    store i16 %live.trunc, ptr %tpidr2.obj.live, align 2
; CHECK-NEXT:    %tpi.int = ptrtoint ptr %tpidr2.call.obj to i64
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 %tpi.int)
; CHECK-NEXT:    call void @new_za_callee()
; CHECK-NEXT:    call void @llvm.aarch64.sme.start.pstateza()
; CHECK-NEXT:    %tpidr2 = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    %cmp = icmp eq i64 %tpidr2, 0
; CHECK-NEXT:    br i1 %cmp, label %restore.za, label %resume
; CHECK:       restore.za:
; CHECK-NEXT:    call void @llvm.aarch64.sme.tpidr2.restore(ptr %tpidr2.call.obj)
; CHECK-NEXT:    br label %resume
; CHECK:       resume:
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    ret void
;
  call void @new_za_callee()
  ret void
}

define void @shared_za_streaming_compatible_caller_private_za_callee() "aarch64_pstate_za_shared" "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_streaming_compatible_caller_private_za_callee() #5 {
; CHECK-NEXT:    %tpidr2.call.obj = alloca %tpidr2_ty, align 8
; CHECK-NEXT:    %N = call i64 @llvm.aarch64.sme.cntsb()
; CHECK-NEXT:    %NN = mul i64 %N, %N
; CHECK-NEXT:    %buffer = alloca i8, i64 %NN, align 16
; CHECK-NEXT:    %tpidr2.call.obj.buffer = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 0
; CHECK-NEXT:    store ptr %buffer, ptr %tpidr2.call.obj.buffer, align 8
; CHECK-NEXT:    %live = call i64 @llvm.aarch64.sme.get.live.za.slices()
; CHECK-NEXT:    %live.trunc = trunc i64 %live to i16
; CHECK-NEXT:    %tpidr2.obj.live = getelementptr %tpidr2_ty, ptr %tpidr2.call.obj, i64 0, i32 1
; CHECK-NEXT:    store i16 %live.trunc, ptr %tpidr2.obj.live, align 2
; CHECK-NEXT:    %tpi.int = ptrtoint ptr %tpidr2.call.obj to i64
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 %tpi.int)
; CHECK-NEXT:    call void @private_za_callee()
; CHECK-NEXT:    call void @llvm.aarch64.sme.start.pstateza()
; CHECK-NEXT:    %tpidr2 = call i64 @llvm.aarch64.sme.get.tpidr2()
; CHECK-NEXT:    %cmp = icmp eq i64 %tpidr2, 0
; CHECK-NEXT:    br i1 %cmp, label %restore.za, label %resume
; CHECK:       restore.za:
; CHECK-NEXT:    call void @llvm.aarch64.sme.tpidr2.restore(ptr %tpidr2.call.obj)
; CHECK-NEXT:    br label %resume
; CHECK:       resume:
; CHECK-NEXT:    call void @llvm.aarch64.sme.set.tpidr2(i64 0)
; CHECK-NEXT:    ret void
;
  call void @private_za_callee()
  ret void
}

; Shared ZA Caller, Shared ZA Callee (Lazy-save not required)

define void @shared_za_shared_za_callee() "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_shared_za_callee() #0 {
; CHECK-NEXT:    call void @shared_za_callee()
; CHECK-NEXT:    ret void
;
  call void @shared_za_callee()
  ret void
}

; Ensure we also check the attribute on the call itself (not just from the called function)
define void @shared_za_shared_za_callee_from_ptr(ptr %fnptr) "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_shared_za_callee_from_ptr(ptr %fnptr) #0 {
; CHECK-NEXT:    call void %fnptr() #0
; CHECK-NEXT:    ret void
;
  call void %fnptr() "aarch64_pstate_za_shared"
  ret void
}

; Shared ZA Caller, Preserved ZA Callee (Lazy-save not required)

define void @shared_za_caller_preserved_za_callee() "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_caller_preserved_za_callee() #0 {
; CHECK-NEXT:    call void @preserved_za_callee()
; CHECK-NEXT:    ret void
;
  call void @preserved_za_callee()
  ret void
}

; Shared ZA Caller with Intrinsic Call (Lazy-save not required)

define float @shared_za_caller_with_intrinsic(ptr %a) "aarch64_pstate_za_shared" {
; CHECK-LABEL: define {{[^@]+}}@shared_za_caller_with_intrinsic(ptr %a) #0 {
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull [[A:%.*]])
; CHECK-NEXT:    [[RES:%.*]] = load float, ptr [[A]], align 4
; CHECK-NEXT:    call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull [[A]])
; CHECK-NEXT:    ret float [[RES]]
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %a)
  %res = load float, ptr %a
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %a)
  ret float %res
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

;.
; CHECK: attributes #0 = { "aarch64_pstate_za_shared" }
; CHECK: attributes #1 = { "aarch64_pstate_za_new" }
; CHECK: attributes #2 = { "aarch64_pstate_za_preserved" }
; CHECK: attributes #3 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
; CHECK: attributes #4 = { "aarch64_expanded_pstate_za" "aarch64_pstate_za_shared" }
; CHECK: attributes #5 = { "aarch64_expanded_pstate_za" "aarch64_pstate_sm_compatible" "aarch64_pstate_za_shared" }
; CHECK: attributes #6 = { argmemonly nocallback nofree nosync nounwind willreturn }
; CHECK: attributes #7 = { nocallback nofree nosync nounwind readnone willreturn }
; CHECK: attributes #8 = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes #9 = { inaccessiblememonly nocallback nofree nosync nounwind readonly willreturn }
;.
