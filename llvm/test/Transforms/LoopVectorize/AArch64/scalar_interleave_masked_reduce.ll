; REQUIRES: asserts
; RUN: opt -S -passes=loop-vectorize -debug-only=loop-vectorize -force-vector-interleave=4 -prefer-predicate-over-epilogue=predicate-dont-vectorize < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; Reproducer for a crash in the VPReductionRecipe's execute method
; when the VF is scalar and the recipe has a mask/condtion.
; Note the "-prefer-predicate-over-epilogue" flag, this is what causes
; the reduction recipe to have a condition. The cost-model will select
; a scalar VF because the target CPU is generic.
; "-force-vector-width=1" cannot be used because the vectorizer will
; generate a different VPlan.

define double @loop(i64 %n, ptr noalias nocapture noundef readonly %c) #0 {

; CHECK: REDUCE ir<%sum.next> = ir<%sum> + reduce.fadd (vp<[[ARG:%.*]]>, vp<[[MASK:%.*]]>)
; CHECK: Executing best plan with VF=1, UF=4

entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %sum = phi double [ 0.000000e+00, %entry ], [ %sum.next, %for.body ]
  %arrayidx1 = getelementptr inbounds [8 x double], ptr %c, i64 0, i64 %i
  %cval = load double, ptr %arrayidx1, align 8
  %sum.next = fadd double %sum, %cval
  %i.next = add nuw nsw i64 %i, 1
  %exitcond.not = icmp eq i64 %i.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  %sum.lcssa = phi double [ %sum.next, %for.body ]
  ret double %sum.lcssa
}

attributes #0 = { nofree nosync nounwind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+crc,+fp-armv8,+lse,+neon,+ras,+rdm,+v8.1a,+v8.2a,+v8a,-fmv" }
