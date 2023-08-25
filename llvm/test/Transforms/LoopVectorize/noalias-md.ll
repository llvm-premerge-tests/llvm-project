; RUN: opt -passes=loop-vectorize -force-vector-width=2 \
; RUN:     -force-vector-interleave=1 -S < %s \
; RUN:     | FileCheck %s -check-prefix=BOTH -check-prefix=LV
; RUN: opt -passes=loop-vectorize,dse -force-vector-width=2 \
; RUN:     -force-vector-interleave=1 -S < %s \
; RUN:     | FileCheck %s -check-prefix=BOTH -check-prefix=DSE

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; This loop needs to be versioned with memchecks between {A, B} x {C} before
; it can be vectorized.
;
;   for (i = 0; i < n; i++) {
;     C[i] = A[i] + 1;
;     C[i] += B[i];
;   }
;
; Check that the corresponding noalias metadata is added to the vector loop
; but not to the scalar loop.
;
; Since in the versioned vector loop C and B can no longer alias, the first
; store to C[i] can be DSE'd.


define void @f(ptr %a, ptr %b, ptr %c) {
entry:
  br label %for.body

; BOTH: vector.memcheck:
; BOTH: vector.body:
for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %inc, %for.body ]

  %arrayidxA = getelementptr inbounds i32, ptr %a, i64 %ind
; Scope 1
; LV: = load {{.*}} !alias.scope [[ALIAS_SCOPE0:![0-9]+]]
  %loadA = load i32, ptr %arrayidxA, align 4

  %add = add nuw i32 %loadA, 2

  %arrayidxC = getelementptr inbounds i32, ptr %c, i64 %ind
; Noalias with scope 1 and 6
; LV: store {{.*}} !alias.scope [[ALIAS_SCOPE1:![0-9]+]], !noalias [[NOALIAS:![0-9]+]]
; DSE-NOT: store
  store i32 %add, ptr %arrayidxC, align 4

  %arrayidxB = getelementptr inbounds i32, ptr %b, i64 %ind
; Scope 6
; LV: = load {{.*}} !alias.scope [[ALIAS_SCOPE2:![0-9]+]]
  %loadB = load i32, ptr %arrayidxB, align 4

  %add2 = add nuw i32 %add, %loadB

; Noalias with scope 1 and 6
; LV: store {{.*}} !alias.scope [[ALIAS_SCOPE1]], !noalias [[NOALIAS]]
; DSE: store
  store i32 %add2, ptr %arrayidxC, align 4

  %inc = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %inc, 20
  br i1 %exitcond, label %for.end, label %for.body

; BOTH: for.body:
; BOTH-NOT: !alias.scope
; BOTH-NOT: !noalias

for.end:                                          ; preds = %for.body
  ret void
}

; LV: [[ALIAS_SCOPE0]] = !{[[DISTINCT0:![0-9]+]]}
; LV: [[DISTINCT0]] = distinct !{[[DISTINCT0]], [[DISTINCT1:![0-9]+]]}
; LV: [[DISTINCT1]] = distinct !{[[DISTINCT1]], !"LVerDomain"}
; LV: [[ALIAS_SCOPE1]] = !{[[DISTINCT2:![0-9]+]]}
; LV: [[DISTINCT2]] = distinct !{[[DISTINCT2]], [[DISTINCT1]]}
; LV: [[NOALIAS]] = !{[[DISTINCT0]], [[DISTINCT3:![0-9]+]]}
; LV: [[DISTINCT3]] = distinct !{[[DISTINCT3]], [[DISTINCT1]]}
; LV: [[ALIAS_SCOPE2]] = !{[[DISTINCT3]]}
