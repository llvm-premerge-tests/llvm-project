; RUN: opt -S -passes=loop-vectorize,dce -mcpu=skylake -vectorizer-ignore-out-of-loop-reduction-cost=0 -force-vector-interleave=1  < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

declare float @llvm.maximum.f32(float, float)
declare float @llvm.fabs.f32(float)

; This is a small trip count loop. The cost of the out-of-loop reduction is
; significant in this case when we only perform a single vector iteration.  
; However, loop vectorizer does not consider out of loop reduction costs.

; CHECK-LABEL: fmaximum_intrinsic
; CHECK-NOT: llvm.vector.reduce.fmaximum 
define float @fmaximum_intrinsic(ptr nocapture readonly %x, ptr nocapture readonly %y, i32 %n, i32 %tc) {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.012 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %s.011 = phi float [ 0.000000e+00, %entry ], [ %max, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %x, i32 %i.012
  %x_f = load float, ptr %arrayidx, align 4
  %arrayidxy = getelementptr inbounds float, ptr %y, i32 %i.012
  %y_f = load float, ptr %arrayidxy, align 4
  %sub = fsub float %x_f, %y_f
  %fabs = call float @llvm.fabs.f32(float %sub)
  %max = tail call float @llvm.maximum.f32(float %s.011, float %fabs)
  %inc = add nuw nsw i32 %i.012, 1
  %exitcond = icmp ult i32 %inc, 3
  br i1 %exitcond, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret float %max
}

; trip count of 6 is still considered non-profitable for reducing adds (min trip
; count required is 8).
; CHECK-LABEL: reduction_sum
; CHECK-NOT: llvm.vector.reduce.add
define i32 @reduction_sum(i32 %n, ptr noalias nocapture %A, ptr noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %5 = load i32, ptr %4, align 4
  %6 = trunc i64 %indvars.iv to i32
  %7 = add i32 %sum.02, %6
  %8 = add i32 %7, %3
  %9 = add i32 %8, %5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph, !prof !1

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}

; CHECK-LABEL: reduction_mix
; CHECK-LABEL: middle.block:
; CHECK-NEXT: vector.reduce.add 
; CHECK-NEXT: br
define i32 @reduction_mix(i32 %n, ptr noalias nocapture %A, ptr noalias nocapture %B) nounwind uwtable readonly noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %sum.02 = phi i32 [ %9, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %3 = load i32, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %5 = load i32, ptr %4, align 4
  %6 = mul nsw i32 %5, %3
  %7 = trunc i64 %indvars.iv to i32
  %8 = add i32 %sum.02, %7
  %9 = add i32 %8, %6
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph, !prof !2

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %0 ], [ %9, %.lr.ph ]
  ret i32 %sum.0.lcssa
}

!1 = !{!"branch_weights", i32 1, i32 5}
!2 = !{!"branch_weights", i32 1, i32 7}
