; RUN: opt -passes=loop-vectorize -force-vector-width=4 -enable-vplan-native-path -S %s | FileCheck %s

; Vectorize explict marked outer loop using vplan native path. Inner loop
; contains simple double add reduction. IR is compiled and modified by hand
; from following C code:
; void inner_loop_reduction(const ptr restrict in_a, const ptr restrict in_b, ptr restrict out)
; {
;     #pragma clang loop vectorize(enable)
;     for (int i = 0; i < 1000; ++i) {
;         double a = in_a[i];
;         double b = in_b[i];
;         for (int j = 0; j < 10000; ++j) {
;             a = a + b;
;         }
;         out[i] = a;
;     }
; }
define void @inner_loop_reduction(ptr noalias nocapture readonly %a.in, ptr noalias nocapture readonly %b.in, ptr noalias nocapture %c.out) {
; CHECK-LABEL: @inner_loop_reduction(

; CHECK: vector.body:
; CHECK-NEXT: %[[FOR1_INDEX:.*]] = phi i64 [ 0, %[[LABEL_PR:.*]] ], [ %{{.*}}, %[[LABEL_FOR1_LATCH:.*]] ]
; CHECK-NEXT: %[[TMP1:.*]] = add i64 %[[FOR1_INDEX]], 0
; CHECK-NEXT: %[[A_PTR:.*]] = getelementptr inbounds double, ptr %a.in, i64 %[[TMP1]]
; CHECK-NEXT: %[[TMP2:.*]] = getelementptr inbounds double, ptr %[[A_PTR]], i32 0
; CHECK-NEXT: %[[WIDE_LOAD1:.*]] = load <4 x double>, ptr %[[TMP2]], align 8
; CHECK-NEXT: %[[B_PTR:.*]] = getelementptr inbounds double, ptr %b.in, i64 %[[TMP1]]
; CHECK-NEXT: %[[TMP3:.*]] = getelementptr inbounds double, ptr %[[B_PTR]], i32 0
; CHECK-NEXT: %[[WIDE_LOAD2:.*]] = load <4 x double>, ptr %[[TMP3]], align 8
; CHECK-NEXT: br label %[[FOR2_HEADER:.*]]

; CHECK: [[FOR2_HEADER]]:
; CHECK-NEXT: %[[FOR2_INDEX:.*]] = phi i32 [ 0, %vector.body ], [ %[[FOR2_INDEX_NEXT:.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[REDUCTION:.*]] = phi <4 x double> [ %[[WIDE_LOAD1]], %vector.body ], [ %[[REDUCTION_NEXT:.*]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[REDUCTION_NEXT]] = fadd <4 x double> %[[WIDE_LOAD2]], %[[REDUCTION]]
; CHECK-NEXT: %[[FOR2_INDEX_NEXT]] = add nuw nsw i32 %[[FOR2_INDEX]], 1
; CHECK-NEXT: %[[EXIT_COND:.*]] = icmp eq i32 %[[FOR2_INDEX_NEXT]], 10000
; CHECK-NEXT: br i1 %[[EXIT_COND]], label %[[FOR1_LATCH:.*]], label %{{.*}}

; CHECK: [[FOR1_LATCH]]:
; CHECK-NEXT: %[[REDUCTION:.*]] = phi <4 x double> [ %[[REDUCTION_NEXT]], %[[FOR2_HEADER]] ]
; CHECK-NEXT: %[[C_PTR:.*]] = getelementptr inbounds double, ptr %c.out, i64 %[[TMP1]]
; CHECK-NEXT: %[[TMP4:.*]] = getelementptr inbounds double, ptr %[[C_PTR]], i32 0
; CHECK-NEXT: store <4 x double> %[[REDUCTION]], ptr %[[TMP4]], align 8
; CHECK-NEXT: %[[FOR1_INDEX_NEXT:.*]] = add nuw i64 %[[FOR1_INDEX]], 4
; CHECK-NEXT: %[[EXIT_COND:.*]] = icmp eq i64 %[[FOR1_INDEX_NEXT]], 1000
; CHECK-NEXT: br i1 %[[EXIT_COND]], label %{{.*}}, label %vector.body

entry:
  br label %for1.header

for1.header:                                              ; preds = %entry
  %indvar1 = phi i64 [ 0, %entry ], [ %indvar11, %for1.latch ]
  %a.ptr = getelementptr inbounds double, ptr %a.in, i64 %indvar1
  %a = load double, ptr %a.ptr, align 8
  %b.ptr = getelementptr inbounds double, ptr %b.in, i64 %indvar1
  %b = load double, ptr %b.ptr, align 8
  br label %for2.header

for2.header:                                              ; preds = %for1.header, %for2.header
  %indvar2 = phi i32 [ 0, %for1.header ], [ %indvar21, %for2.header ]
  %a.reduction = phi double [ %a, %for1.header ], [ %a.reduction1, %for2.header ]
  %a.reduction1 = fadd double %b, %a.reduction
  %indvar21 = add nuw nsw i32 %indvar2, 1
  %for2.cond = icmp eq i32 %indvar21, 10000
  br i1 %for2.cond, label %for1.latch, label %for2.header

for1.latch:                                               ; preds = %for2.header
  %c.ptr = getelementptr inbounds double, ptr %c.out, i64 %indvar1
  store double %a.reduction1, ptr %c.ptr, align 8
  %indvar11 = add nuw nsw i64 %indvar1, 1
  %for1.cond = icmp eq i64 %indvar11, 1000
  br i1 %for1.cond, label %exit, label %for1.header, !llvm.loop !0

exit:                                                    ; preds = %for1.latch
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
