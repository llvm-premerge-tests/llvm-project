; REQUIRES: x86

; RUN: rm -rf %t.dir
; RUN: split-file %s %t.dir
; RUN: cd %t.dir

;; Common artifacts
; RUN: opt --thinlto-bc -o %t1.o ThinLTO.ll
; RUN: opt -o %t2_regular.o RegularLTO.ll
; RUN: opt -module-summary -o %t2_summary.o RegularLTO.ll

;; --lto-whole-program-visibility when there's ThinLTO and a RegularLTO without summary causes RegularLTO
;; to optimize itself first which exports the resolution to the combined index. However, ThinLTO
;; is unaware of RegularLTO types and will optimize separately.
; RUN: ld.lld %t1.o %t2_regular.o -o %t3 -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=BASE
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-IR,CHECK-COMMON-IR
; RUN: llvm-dis %t3.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-REGULAR-IR,CHECK-COMMON-REGULAR-IR

;; RegularLTO decision
; BASE-DAG: single-impl: devirtualized a call to _ZN7Regular1fEi
; BASE-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; BASE-DAG: single-impl: devirtualized a call to _ZN1D1mEi
;; ThinLTO summary decision
; BASE-DAG: single-impl: devirtualized a call to _ZN1A1fEi

;; --lto-whole-program-visibility when there's ThinLTO and a RegularLTO with summary optimizes
;; using the combined index and interact with the other. However, RegularLTO calculates its own
;; resolutions that will be used by ThinLTO as well
; RUN: ld.lld %t1.o %t2_summary.o -o %t4 -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=SUMMARY
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-SUMMARY-IR,CHECK-COMMON-IR
; RUN: llvm-dis %t4.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-SUMMARY-REGULAR-IR,CHECK-COMMON-REGULAR-IR

;; RegularLTO decision
; SUMMARY-DAG: single-impl: devirtualized a call to _ZN7Regular1fEi
; SUMMARY-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; SUMMARY-DAG: single-impl: devirtualized a call to _ZN1D1mEi
;; ThinLTO identifies this as not singleimpl but still applies RegularLTO resolution
; SUMMARY-DAG: single-impl: devirtualized a call to _ZN7Regular1fEi

;; --lto-validate-all-vtables-have-type-infos when there's ThinLTO and a RegularLTO without summary identifies that
;; types cross the LTO boundary and prevents visibility upgrades with --lto-whole-program-visibility
; RUN: ld.lld %t1.o %t2_regular.o -o %t5 -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-VALIDATE-IR,CHECK-COMMON-IR
; RUN: llvm-dis %t5.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-VALIDATE-REGULAR-IR,CHECK-COMMON-REGULAR-IR

; VALIDATE-DAG: single-impl: devirtualized a call to _ZN1D1mEi

;; --lto-validate-all-vtables-have-type-infos when there's ThinLTO and a RegularLTO with summary
;; means its summary is added to the combined index and devirt only occurs in the combined index
; RUN: ld.lld %t1.o %t2_summary.o -o %t6 -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=VALIDATE-SUMMARY
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-VALIDATE-SUMMARY-IR,CHECK-COMMON-IR
; RUN: llvm-dis %t6.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-VALIDATE-SUMMARY-REGULAR-IR,CHECK-COMMON-REGULAR-IR

; VALIDATE-SUMMARY-DAG: single-impl: devirtualized a call to _ZN1A1nEi
; VALIDATE-SUMMARY-DAG: single-impl: devirtualized a call to _ZN1D1mEi

;--- ThinLTO.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { ptr }

;; Placeholders for testing and also check that these don't trigger --lto-validate-all-vtables-have-type-infos
@_ZTVN10__cxxabiv117__class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }
@_ZTVN10__cxxabiv120__si_class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }

@_ZTV1B = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1A1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5
@_ZTV1C = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1A1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1, !type !2, !type !6, !type !7, !type !8
@_ZTV1D = internal constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1D1mEi] }, !type !9, !vcall_visibility !11

@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }

@_ZTS1B = linkonce_odr constant [3 x i8] c"1B\00"
@_ZTI1B = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS1B, ptr @_ZTI1A }

@_ZTS1C = linkonce_odr constant [3 x i8] c"1C\00"
@_ZTI1C = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS1C, ptr @_ZTI1A }

@_ZTS1D = internal constant [3 x i8] c"1D\00"
@_ZTI1D = internal constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1D }

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [3 x ptr] [ ptr @_ZTV1B, ptr @_ZTV1C, ptr @_ZTV1D ]

; CHECK-COMMON-IR-LABEL: define dso_local i32 @_start
define i32 @_start(ptr %obj, ptr %obj2, ptr %obj3, i32 %a) {
  ;; Call function built with RegularLTO
  %RegularLTOResult = call i32 @RegularLTO(ptr %obj, i32 %a)

  ;; ThinLTO code starts here
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  ; CHECK-SUMMARY-IR: %call = tail call i32 @_ZN1A1nEi
  ;; Not devirtualized because of crossing LTO boundary
  ; CHECK-VALIDATE-IR: %call = tail call i32 %fptr1
  ;; With everything present in summary, crossing LTO boundary is okay
  ; CHECK-VALIDATE-SUMMARY-IR: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  %fptr22 = load ptr, ptr %vtable, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call2 = tail call i32 @_ZN1A1fEi
  ;; Summary analysis does not devirtualize this, but RegularLTO's
  ;; resolution still applies.
  ; CHECK-SUMMARY-IR: %call2 = tail call i32 @_ZN7Regular1fEi
  ;; Not devirtualized because of crossing LTO boundary
  ; CHECK-VALIDATE-IR: %call2 = tail call i32 %fptr22
  ;; With everything present in summary, identified as not singleimpl
  ; CHECK-VALIDATE-SUMMARY-IR: %call2 = tail call i32 %fptr22
  %call2 = tail call i32 %fptr22(ptr nonnull %obj, i32 %call)

  %vtable2 = load ptr, ptr %obj2
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !10)
  call void @llvm.assume(i1 %p2)

  %fptr33 = load ptr, ptr %vtable2, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call3 = tail call i32 @_ZN1D1mEi
  ; CHECK-SUMMARY-IR: %call3 = tail call i32 @_ZN1D1mEi
  ; CHECK-VALIDATE-IR: %call3 = tail call i32 @_ZN1D1mEi
  ; CHECK-VALIDATE-SUMMARY-IR: %call3 = tail call i32 @_ZN1D1mEi
  %call3 = tail call i32 %fptr33(ptr nonnull %obj2, i32 %call2)

  ret i32 %call3
}
; CHECK-COMMON-IR-LABEL: ret i32
; CHECK-COMMON-IR-LABEL: }

declare i32 @RegularLTO(ptr)
declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define linkonce_odr i32 @_ZN1A1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define linkonce_odr i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define internal i32 @_ZN1D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFviE.virtual"}
!2 = !{i64 24, !"_ZTSM1AFviE.virtual"}
!3 = !{i64 16, !"_ZTS1B"}
!4 = !{i64 16, !"_ZTSM1BFviE.virtual"}
!5 = !{i64 24, !"_ZTSM1BFviE.virtual"}
!6 = !{i64 16, !"_ZTS1C"}
!7 = !{i64 16, !"_ZTSM1CFviE.virtual"}
!8 = !{i64 24, !"_ZTSM1CFviE.virtual"}
!9 = !{i64 16, !10}
!10 = distinct !{}
!11 = !{i64 2}

;--- RegularLTO.ll
; REQUIRES: x86

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.Native = type { %struct.A }

; Placeholders for testing
@_ZTVN10__cxxabiv117__class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }
@_ZTVN10__cxxabiv120__si_class_type_infoE = linkonce_odr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr null] }

@_ZTV7Regular = linkonce_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI7Regular, ptr @_ZN7Regular1fEi, ptr @_ZN1A1nEi] } , !type !0, !type !1, !type !2, !type !3, !type !4, !type !5
@_ZTS7Regular = linkonce_odr constant [9 x i8] c"7Regular\00"
@_ZTI7Regular = linkonce_odr constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS7Regular, ptr @_ZTI1A }

; Base type A does not need to emit a vtable if it's never instantiated. However, RTTI still gets generated
@_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00"
@_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [1 x ptr] [ ptr @_ZTV7Regular ]

; CHECK-COMMON-REGULAR-IR-LABEL: define dso_local i32 @RegularLTO
define i32 @RegularLTO(ptr %obj, i32 %a) #0 {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 0
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-REGULAR-IR: %call = tail call i32 @_ZN7Regular1fEi
  ; CHECK-SUMMARY-REGULAR-IR: %call = tail call i32 @_ZN7Regular1fEi
  ;; Not devirtualized because of crossing LTO boundary
  ; CHECK-VALIDATE-REGULAR-IR: %call = tail call i32 %fptr1
  ;; With everything present in summary, identified as not singleimpl
  ; CHECK-VALIDATE-SUMMARY-REGULAR-IR: %call = tail call i32 %fptr1
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-COMMON-REGULAR-IR-LABEL: ret i32
; CHECK-COMMON-REGULAR-IR-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define linkonce_odr i32 @_ZN7Regular1fEi(ptr %this, i32 %a) #0 {
   ret i32 1;
}

define linkonce_odr i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }
!llvm.module.flags = !{!6}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFviE.virtual"}
!2 = !{i64 24, !"_ZTSM1AFviE.virtual"}
!3 = !{i64 16, !"_ZTS7Regular"}
!4 = !{i64 16, !"_ZTSM7RegularFviE.virtual"}
!5 = !{i64 24, !"_ZTSM7RegularFviE.virtual"}
!6 = !{i32 1, !"ThinLTO", i32 0}
