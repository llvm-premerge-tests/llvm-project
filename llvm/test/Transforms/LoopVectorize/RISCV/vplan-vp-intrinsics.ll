; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=if-explicit-vector-length-support \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output %s 2>&1 | FileCheck --check-prefixes=IF-EVL,CHECK %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=force-explicit-vector-length-support \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output %s 2>&1 | FileCheck --check-prefixes=FORCE-EVL,CHECK %s

; RUN: opt -passes=loop-vectorize -debug-only=loop-vectorize \
; RUN: -prefer-predicate-with-vp-intrinsics=no-predication \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -disable-output %s 2>&1 | FileCheck --check-prefixes=NO-VP,CHECK %s

define void @foo(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %N) {
; IF-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; IF-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; IF-EVL-NEXT: Live-in ir<%N> = original trip-count
; IF-EVL-EMPTY:
; IF-EVL:      vector.ph:
; IF-EVL-NEXT: Successor(s): vector loop
; IF-EVL-EMPTY:
; IF-EVL-NEXT: <x1> vector loop: {
; IF-EVL-NEXT:  vector.body:
; IF-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; IF-EVL-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; IF-EVL-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[IV]]>, vp<[[VTC]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = load ir<[[GEP1]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; IF-EVL-NEXT:    WIDEN ir<[[LD2:%.+]]> = load ir<[[GEP2]]>
; IF-EVL-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; IF-EVL-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; IF-EVL-NEXT:    WIDEN store ir<[[GEP3]]>, ir<[[ADD]]>
; IF-EVL-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = VF * UF +  vp<[[IV]]>, vp<[[EVL]]>
; IF-EVL-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; IF-EVL-NEXT:  No successors
; IF-EVL-NEXT: }

; FORCE-EVL: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; FORCE-EVL-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; FORCE-EVL-NEXT: Live-in ir<%N> = original trip-count
; FORCE-EVL-EMPTY:
; FORCE-EVL:      vector.ph:
; FORCE-EVL-NEXT: Successor(s): vector loop
; FORCE-EVL-EMPTY:
; FORCE-EVL-NEXT: <x1> vector loop: {
; FORCE-EVL-NEXT:  vector.body:
; FORCE-EVL-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; FORCE-EVL-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; FORCE-EVL-NEXT:    EMIT vp<[[EVL:%.+]]> = EXPLICIT-VECTOR-LENGTH vp<[[IV]]>, vp<[[VTC]]>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN ir<[[LD1:%.+]]> = load ir<[[GEP1]]>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN ir<[[LD2:%.+]]> = load ir<[[GEP2]]>
; FORCE-EVL-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; FORCE-EVL-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; FORCE-EVL-NEXT:    WIDEN store ir<[[GEP3]]>, ir<[[ADD]]>
; FORCE-EVL-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = VF * UF +  vp<[[IV]]>, vp<[[EVL]]>
; FORCE-EVL-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; FORCE-EVL-NEXT:  No successors
; FORCE-EVL-NEXT: }

; NO-VP: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2,vscale x 4},UF>=1' {
; NO-VP-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; NO-VP-NEXT: Live-in ir<%N> = original trip-count
; NO-VP-EMPTY:
; NO-VP:      vector.ph:
; NO-VP-NEXT: Successor(s): vector loop
; NO-VP-EMPTY:
; NO-VP-NEXT: <x1> vector loop: {
; NO-VP-NEXT:  vector.body:
; NO-VP-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; NO-VP-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; NO-VP-NEXT:    EMIT vp<[[MASK:%.+]]> = active lane mask vp<[[ST]]>, ir<%N>
; NO-VP-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr inbounds ir<%b>, vp<[[ST]]>
; NO-VP-NEXT:    WIDEN ir<[[LD1:%.+]]> = load ir<[[GEP1]]>
; NO-VP-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr inbounds ir<%c>, vp<[[ST]]>
; NO-VP-NEXT:    WIDEN ir<[[LD2:%.+]]> = load ir<[[GEP2]]>
; NO-VP-NEXT:    WIDEN ir<[[ADD:%.+]]> = add nsw ir<[[LD2]]>, ir<[[LD1]]>
; NO-VP-NEXT:    CLONE ir<[[GEP3:%.+]]> = getelementptr inbounds ir<%a>, vp<[[ST]]>
; NO-VP-NEXT:    WIDEN store ir<[[GEP3]]>, ir<[[ADD]]>
; NO-VP-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = VF * UF + vp<[[IV]]>
; NO-VP-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; NO-VP-NEXT:  No successors
; NO-VP-NEXT: }

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

define void @safe_dep(ptr %p) {
; CHECK: VPlan 'Initial VPlan for VF={vscale x 1,vscale x 2},UF>=1' {
; CHECK-NEXT: Live-in vp<[[VTC:%[0-9]+]]> = vector-trip-count
; CHECK-NEXT: Live-in ir<512> = original trip-count
; CHECK-EMPTY:
; CHECK:      vector.ph:
; CHECK-NEXT: Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT: <x1> vector loop: {
; CHECK-NEXT:  vector.body:
; CHECK-NEXT:    EMIT vp<[[IV:%[0-9]+]]> = CANONICAL-INDUCTION
; CHECK-NEXT:    vp<[[ST:%[0-9]+]]>    = SCALAR-STEPS vp<[[IV]]>, ir<1>
; CHECK-NEXT:    CLONE ir<[[GEP1:%.+]]> = getelementptr ir<%p>, vp<[[ST]]>
; CHECK-NEXT:    WIDEN ir<[[V:%.+]]> = load ir<[[GEP1]]>
; CHECK-NEXT:    CLONE ir<[[OFFSET:.+]]> = add vp<[[ST]]>, ir<100>
; CHECK-NEXT:    CLONE ir<[[GEP2:%.+]]> = getelementptr ir<%p>, ir<[[OFFSET]]>
; CHECK-NEXT:    WIDEN store ir<[[GEP2]]>, ir<[[V]]>
; CHECK-NEXT:    EMIT vp<[[IV_NEXT:%[0-9]+]]> = VF * UF + nuw vp<[[IV]]>
; CHECK-NEXT:    EMIT branch-on-count  vp<[[IV_NEXT]]>, vp<[[VTC]]>
; CHECK-NEXT:  No successors
; CHECK-NEXT: }

entry:
  br label %loop

loop:
  %iv = phi i64 [0, %entry], [%iv.next, %loop]
  %a1 = getelementptr i64, ptr %p, i64 %iv
  %v = load i64, ptr %a1, align 32
  %offset = add i64 %iv, 100
  %a2 = getelementptr i64, ptr %p, i64 %offset
  store i64 %v, ptr %a2, align 32
  %iv.next = add i64 %iv, 1
  %cmp = icmp ne i64 %iv, 511
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

