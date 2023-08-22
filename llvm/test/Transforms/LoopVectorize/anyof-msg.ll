; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s -debug-only=iv-descriptors,loop-vectorize -S 2>&1 | FileCheck %s

; The test case is copied from Transforms/LoopVectorize/select-cmp.ll
;
; CHECK-LABEL: Checking a loop in 'select_i32_from_icmp_same_inputs'
; CHECK: Found a float conditional select reduction PHI. PHI.  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
;
define i32 @select_i32_from_icmp_same_inputs(i32 %a, i32 %b, i64 %n) {
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %4, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %3, %for.body ]
  %2 = icmp eq i32 %1, 3
  %3 = select i1 %2, i32 %1, i32 %b
  %4 = add nuw nsw i64 %0, 1
  %5 = icmp eq i64 %4, %n
  br i1 %5, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %3
}

;
; The test case is modified from Transforms/LoopVectorize/select-cmp.ll
;
; CHECK-LABEL: Checking a loop in 'select_i32_from_icmp_non_const_same_inputs'
; CHECK: Found a float conditional select reduction PHI. PHI.  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
;
define i32 @select_i32_from_icmp_non_const_same_inputs(i32 %a, i32 %b, i64 %n, ptr %c) {
entry:
  br label %for.body

for.body:                                      ; preds = %entry, %for.body
  %0 = phi i64 [ 0, %entry ], [ %6, %for.body ]
  %1 = phi i32 [ %a, %entry ], [ %5, %for.body ]
  %2 = getelementptr inbounds i32, ptr %c, i64 %0
  %3 = load i32, ptr %2, align 4
  %4 = icmp eq i32 %1, %3
  %5 = select i1 %4, i32 %1, i32 %b
  %6 = add nuw nsw i64 %0, 1
  %7 = icmp eq i64 %6, %n
  br i1 %7, label %exit, label %for.body

exit:                                     ; preds = %for.body
  ret i32 %5
}
