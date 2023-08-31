; RUN: opt -passes=loop-vectorize -force-vector-width=4 -enable-vplan-native-path -S %s | FileCheck %s

; Test that VPlan native path is able to widen call intructions like
; llvm.sqrt.* intrincis calls.

declare double @llvm.sqrt.f64(double %0)
define void @widen_call_instruction(ptr noalias nocapture readonly %a.in, ptr noalias nocapture readonly %b.in, ptr noalias nocapture %c.out) {
; CHECK-LABEL: @widen_call_instruction(


; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[FOR1_LATCH:.*]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds double, ptr %a.in, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds double, ptr [[TMP1]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x double>, ptr [[TMP2]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds double, ptr %b.in, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds double, ptr [[TMP3]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD1:%.*]] = load <4 x double>, ptr [[TMP4]], align 8
; CHECK-NEXT:    [[TMP5:%.*]] = call <4 x double> @llvm.sqrt.v4f64(<4 x double> [[WIDE_LOAD1]])
; CHECK-NEXT:    br label %[[FOR2_HEADER:.*]]

; CHECK:       [[FOR2_HEADER]]:
; CHECK-NEXT:    [[SCALAR_PHI:%.*]] = phi i32 [ 0, %vector.body ], [ [[TMP7:%.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT:    [[VEC_PHI:%.*]] = phi <4 x double> [ [[WIDE_LOAD]], %vector.body ], [ [[TMP6:%.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT:    [[TMP6]] = fadd <4 x double> [[TMP5]], [[VEC_PHI]]
; CHECK-NEXT:    [[TMP7]] = add nuw nsw i32 [[SCALAR_PHI]], 1
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i32 [[TMP7]], 10000
; CHECK-NEXT:    br i1 [[TMP8]], label %[[FOR1_LATCH]], label %[[FOR2_HEADER]]

; CHECK:       [[FOR1_LATCH]]:
; CHECK-NEXT:    [[VEC_PHI4:%.*]] = phi <4 x double> [ [[TMP6]], %[[FOR2_HEADER]] ]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds double, ptr %c.out, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds double, ptr [[TMP9]], i32 0
; CHECK-NEXT:    store <4 x double> [[VEC_PHI4]], ptr [[TMP10]], align 8
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP11:%.*]] = icmp eq i64 [[INDEX_NEXT]], 1000
; CHECK-NEXT:    br i1 [[TMP11]], label [[MIDDLE_BLOCK:%.*]], label %vector.body

entry:
  br label %for1.header

for1.header:
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %a.ptr = getelementptr inbounds double, ptr %a.in, i64 %indvar1
  %a = load double, ptr %a.ptr, align 8
  %b.ptr = getelementptr inbounds double, ptr %b.in, i64 %indvar1
  %b = load double, ptr %b.ptr, align 8
  %b.sqrt = call double @llvm.sqrt.f64(double %b)
  br label %for2.header

for2.header:
  %indvar2 = phi i32 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %a.reduction = phi double [ %a, %for1.header ], [ %a.reduction1, %for2.header ]
  %a.reduction1 = fadd double %b.sqrt, %a.reduction
  %indvar21 = add nuw nsw i32 %indvar2, 1
  %for2.cond = icmp eq i32 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:
  %c.ptr = getelementptr inbounds double, ptr %c.out, i64 %indvar1
  store double %a.reduction1, ptr %c.ptr, align 8
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
