; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=if-explicit-vector-length-support \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=force-explicit-vector-length-support \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck --check-prefixes=FORCE-EVL %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=no-predication \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -force-target-supports-scalable-vectors -scalable-vectorization=on \
; RUN: -disable-output < %s 2>&1 | FileCheck --check-prefixes=NO-VP %s

define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; FORCE-EVL: VPlan 'Initial VPlan for VF={vscale x 1},UF>=1' {
; FORCE-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; FORCE-EVL-NEXT: Live-in ir<%N> = original trip-count
; FORCE-EVL-EMPTY:
; FORCE-EVL:      vector.ph:
; FORCE-EVL-NEXT: Successor(s): vector loop
; FORCE-EVL-EMPTY:
; FORCE-EVL-NEXT: <x1> vector loop: {
; FORCE-EVL-NEXT:  vector.body:
; FORCE-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; FORCE-EVL-NEXT:    EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI vp<[[EVL_PHI:%[0-9]+]]> = phi ir<0>, vp<[[IV_NEXT:%[0-9]+]]>
; FORCE-EVL-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[EVL_PHI]]>, ir<%N>
; FORCE-EVL-NEXT:    vp<[[ST:%[0-9]+]]> = SCALAR-STEPS vp<[[EVL_PHI]]>, ir<1>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = load ir<[[GEP1]]>, ir<true>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN ir<[[LD2:%.+]]> = load ir<[[GEP2]]>, ir<true>
; FORCE-EVL-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN store ir<[[GEP3]]>, ir<[[ADD]]>, ir<true>
; FORCE-EVL-NEXT:    EMIT vp<[[IV_NEXT]]> = EXPLICIT-VECTOR-LENGTH + vp<[[EVL_PHI]]>, vp<[[EVL]]>
; FORCE-EVL-NEXT:    EMIT vp<[[IV_NEXT_EXIT:%[0-9]+]]> = VF * UF + vp<[[IV]]>
; FORCE-EVL-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT_EXIT]]>, vp<[[VTC]]>
; FORCE-EVL-NEXT:  No successors
; FORCE-EVL-NEXT: }

; NO-VP-NOT: EXPLICIT-VECTOR-LENGTH-BASED-IV-PHI

entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

