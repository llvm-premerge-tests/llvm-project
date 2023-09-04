; RUN: opt -S -passes=bpf-hoist-argument-access -mtriple=bpf-pc-linux < %s | FileCheck %s

%struct.foo = type { i32, i32, i32 }
declare void @consume(i32 noundef)
declare i32 @produce()
@gptr = external global ptr
@gptr2 = external global ptr

define void @load.1(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @load.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[T1:%.*]] = load i32, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[V2:%.*]] = phi i32 [ [[T0]], %if.then ], [ [[T1]], %if.else ]
; CHECK-NEXT:    call void @consume(i32 [[V2]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @load.one_hand_nonconst.1(ptr %ctx, i1 %cond, i32 %idx) {
; CHECK-LABEL: define void @load.one_hand_nonconst.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]], i32 [[IDX:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 [[IDX]], i32 1
; CHECK-NEXT:    [[T1:%.*]] = load i32, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[V2:%.*]] = phi i32 [ [[T0]], %if.then ], [ [[T1]], %if.else ]
; CHECK-NEXT:    call void @consume(i32 [[V2]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 %idx, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @load.one_hand_nonarg.1(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @load.one_hand_nonarg.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 0
; CHECK-NEXT:    [[T1:%.*]] = load i32, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[V2:%.*]] = phi i32 [ [[T1]], %if.then ], [ [[T0]], %if.else ]
; CHECK-NEXT:    call void @consume(i32 [[V2]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @load.one_hand_many_succ.1(ptr %ctx, i1 %cond, i1 %cond2) {
; CHECK:       define void @load.one_hand_many_succ.1(ptr %[[ctx:.*]], i1 %[[cond:.*]], i1 %[[cond2:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %[[cond]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    br i1 %[[cond]], label %if.end.sa.lh, label %if.else
; CHECK:       if.else:
; CHECK-NEXT:    %[[p2:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i32 0, i32 1
; CHECK-NEXT:    %[[T0:.*]] = load i32, ptr %[[p2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end.sa.lh:
; CHECK-NEXT:    %[[p11:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i32 0, i32 0
; CHECK-NEXT:    %[[T1:.*]] = load i32, ptr %[[p11]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    %[[v3:.*]] = phi i32 [ %[[T0]], %if.else ],
; CHECK-SAME:                         [ %[[T1]], %if.end.sa.lh ]
; CHECK-NEXT:    call void @consume(i32 %[[v3]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br i1 %cond, label %if.end, label %if.else
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @noop.1(i1 %cond) {
; CHECK-LABEL: define void @noop.1
; CHECK-SAME: (i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 0
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 1
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[P:%.*]] = phi ptr [ [[P1]], %if.then ], [ [[P2]], %if.else ]
; CHECK-NEXT:    [[V:%.*]] = load i32, ptr [[P]], align 4
; CHECK-NEXT:    call void @consume(i32 [[V]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @noop.2(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @noop.2
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[P:%.*]] = phi ptr [ [[P1]], %if.then ], [ [[P2]], %if.else ]
; CHECK-NEXT:    br label %if.end.2
; CHECK:       if.end.2:
; CHECK-NEXT:    [[V:%.*]] = load i32, ptr [[P]], align 4
; CHECK-NEXT:    call void @consume(i32 [[V]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  br label %if.end.2
if.end.2:
  %v = load i32, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @store.1(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @store.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    store i32 42, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    store i32 42, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  store i32 42, ptr %p
  ret void
}

define void @two_times.no_uses_after.1(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @two_times.no_uses_after.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[P1]], align 4
; CHECK-NEXT:    store i32 [[T0]], ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[T1:%.*]] = load i32, ptr [[P2]], align 4
; CHECK-NEXT:    store i32 [[T1]], ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  store i32 %v, ptr %p
  ret void
}

define void @two_times.with_use_after.2(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @two_times.with_use_after.2
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[T0:%.*]] = load i32, ptr [[P1]], align 4
; CHECK-NEXT:    store i32 [[T0]], ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[T1:%.*]] = load i32, ptr [[P2]], align 4
; CHECK-NEXT:    store i32 [[T1]], ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[V2:%.*]] = phi i32 [ [[T0]], %if.then ], [ [[T1]], %if.else ]
; CHECK-NEXT:    call void @consume(i32 [[V2]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %v = load i32, ptr %p
  store i32 %v, ptr %p
  call void @consume(i32 %v)
  ret void
}

define void @phi.1(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @phi.1
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[A:%.*]] = load i32, ptr @gptr, align 4
; CHECK-NEXT:    store i32 [[A]], ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[B:%.*]] = load i32, ptr @gptr2, align 4
; CHECK-NEXT:    store i32 [[B]], ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  %a = load i32, ptr @gptr
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  %b = load i32, ptr @gptr2
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %ab = phi i32 [%a, %if.then], [%b, %if.else]
  store i32 %ab, ptr %p
  ret void
}

define void @phi.2(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @phi.2
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[A:%.*]] = load i32, ptr @gptr, align 4
; CHECK-NEXT:    store i32 42, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[B:%.*]] = load i32, ptr @gptr2, align 4
; CHECK-NEXT:    store i32 42, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    [[AB:%.*]] = phi i32 [ [[A]], %if.then ], [ [[B]], %if.else ]
; CHECK-NEXT:    call void @consume(i32 [[AB]])
; CHECK-NEXT:    ret void
;
entry:
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  %a = load i32, ptr @gptr
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  %b = load i32, ptr @gptr2
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %ab = phi i32 [%a, %if.then], [%b, %if.else]
  store i32 42, ptr %p
  call void @consume(i32 %ab)
  ret void
}

define void @phi.3(ptr %ctx, i1 %cond) {
; CHECK-LABEL: define void @phi.3
; CHECK-SAME: (ptr [[CTX:%.*]], i1 [[COND:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[K:%.*]] = call i32 @produce()
; CHECK-NEXT:    br i1 [[COND]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    [[A:%.*]] = load i32, ptr @gptr, align 4
; CHECK-NEXT:    call void @consume(i32 [[K]])
; CHECK-NEXT:    store i32 42, ptr [[P1]], align 4
; CHECK-NEXT:    call void @consume(i32 [[A]])
; CHECK-NEXT:    store i32 7, ptr [[P1]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    [[B:%.*]] = load i32, ptr @gptr2, align 4
; CHECK-NEXT:    call void @consume(i32 [[K]])
; CHECK-NEXT:    store i32 42, ptr [[P2]], align 4
; CHECK-NEXT:    call void @consume(i32 [[B]])
; CHECK-NEXT:    store i32 7, ptr [[P2]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret void
;
entry:
  %k = call i32 @produce()
  br i1 %cond, label %if.then, label %if.else
if.then:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  %a = load i32, ptr @gptr
  br label %if.end
if.else:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  %b = load i32, ptr @gptr2
  br label %if.end
if.end:
  %p = phi ptr [%p1, %if.then], [%p2, %if.else]
  %ab = phi i32 [%a, %if.then], [%b, %if.else]
  ;; value 'k' does not depend on phi or GEP chains that would be
  ;; lifted, check that implementation handles such cases.
  call void @consume(i32 %k)
  store i32 42, ptr %p
  call void @consume(i32 %ab)
  store i32 7, ptr %p
  ret void
}

define void @switch.1(ptr %ctx, i32 %smth) {
; CHECK-LABEL: define void @switch.1
; CHECK-SAME: (ptr [[CTX:%.*]], i32 [[SMTH:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 [[SMTH]], label [[EXIT:%.*]] [
; CHECK-NEXT:    i32 1, label [[BB1:%.*]]
; CHECK-NEXT:    i32 2, label [[BB2:%.*]]
; CHECK-NEXT:    i32 3, label [[BB3:%.*]]
; CHECK-NEXT:    i32 4, label [[BB4:%.*]]
; CHECK-NEXT:    ]
; CHECK:       bb1:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 0
; CHECK-NEXT:    store i32 42, ptr [[P1]], align 4
; CHECK-NEXT:    br label %end
; CHECK:       bb2:
; CHECK-NEXT:    [[P2:%.*]] = getelementptr inbounds %struct.foo, ptr [[CTX]], i32 0, i32 1
; CHECK-NEXT:    store i32 42, ptr [[P2]], align 4
; CHECK-NEXT:    br label %end
; CHECK:       bb3:
; CHECK-NEXT:    [[P3:%.*]] = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 1
; CHECK-NEXT:    br label %end.lh
; CHECK:       bb4:
; CHECK-NEXT:    [[P4:%.*]] = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 2
; CHECK-NEXT:    br label %end.lh
; CHECK:       end.lh:
; CHECK-NEXT:    [[P_PH:%.*]] = phi ptr [ [[P4]], [[BB4]] ], [ [[P3]], [[BB3]] ]
; CHECK-NEXT:    store i32 42, ptr [[P_PH]], align 4
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  switch i32 %smth, label %exit [
  i32 1, label %bb1
  i32 2, label %bb2
  i32 3, label %bb3
  i32 4, label %bb4
  ]
bb1:
  %p1 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 0
  br label %end
bb2:
  %p2 = getelementptr inbounds %struct.foo, ptr %ctx, i32 0, i32 1
  br label %end
bb3:
  %p3 = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 1
  br label %end
bb4:
  %p4 = getelementptr inbounds %struct.foo, ptr @gptr, i32 0, i32 2
  br label %end
end:
  %p = phi ptr [%p1, %bb1], [%p2, %bb2], [%p3, %bb3], [%p4, %bb4]
  store i32 42, ptr %p
  br label %exit
exit:
  ret void
}
