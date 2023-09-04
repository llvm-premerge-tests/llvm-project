; RUN: opt -S -passes=bpf-hoist-argument-access -mtriple=bpf-pc-linux < %s | FileCheck %s

%struct.foo = type { i32, i32 }
declare void @consume(i32)
declare void @consume_ptr(ptr)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr)

@"llvm.bpf_sockopt:0:20$0:3" = external global i64, !llvm.preserve.access.index !0

define void @select.1(ptr %ctx, i1 %cond) {
; CHECK:       define void @select.1(ptr %[[ctx:.*]], i1 %[[cond:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %[[cond]], label %entry.select.true, label %entry.select.false
; CHECK:       entry.select.true:
; CHECK-NEXT:    %[[a2:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 0
; CHECK-NEXT:    %[[T0:.*]] = load i32, ptr %[[a2]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.false:
; CHECK-NEXT:    %[[b3:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 1
; CHECK-NEXT:    %[[T1:.*]] = load i32, ptr %[[b3]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.tail:
; CHECK-NEXT:    %[[v5:.*]] = phi i32 [ %[[T0]], %entry.select.true ],
; CHECK-SAME:                         [ %[[T1]], %entry.select.false ]
; CHECK-NEXT:    call void @consume(i32 %[[v5]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %a = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 0
  %b = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 1
  %p = select i1 %cond, ptr %a, ptr %b
  %v = load i32, ptr %p, align 4
  call void @consume(i32 %v)
  ret void
}

define void @select.one-hand.1(ptr %ctx, i1 %cond, i64 %idx) {
; CHECK:       define void @select.one-hand.1(ptr %[[ctx:.*]], i1 %[[cond:.*]], i64 %[[idx:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[b:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 %[[idx]], i32 1
; CHECK-NEXT:    br i1 %[[cond]], label %entry.select.true, label %entry.select.false
; CHECK:       entry.select.true:
; CHECK-NEXT:    %[[a2:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 0
; CHECK-NEXT:    %[[T0:.*]] = load i32, ptr %[[a2]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.false:
; CHECK-NEXT:    %[[T1:.*]] = load i32, ptr %[[b]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.tail:
; CHECK-NEXT:    %[[v4:.*]] = phi i32 [ %[[T0]], %entry.select.true ],
; CHECK-SAME:                         [ %[[T1]], %entry.select.false ]
; CHECK-NEXT:    call void @consume(i32 %[[v4]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %a = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 0
  %b = getelementptr inbounds %struct.foo, ptr %ctx, i64 %idx, i32 1
  %p = select i1 %cond, ptr %a, ptr %b
  %v = load i32, ptr %p, align 4
  call void @consume(i32 %v)
  ret void
}

define void @select.use_after.1(ptr %ctx, i1 %cond) {
; CHECK:       define void @select.use_after.1(ptr %[[ctx:.*]], i1 %[[cond:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[a:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 0
; CHECK-NEXT:    %[[b:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 1
; CHECK-NEXT:    br i1 %[[cond]], label %entry.select.true, label %entry.select.false
; CHECK:       entry.select.true:
; CHECK-NEXT:    %[[a2:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 0
; CHECK-NEXT:    %[[T0:.*]] = load i32, ptr %[[a2]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.false:
; CHECK-NEXT:    %[[b3:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 1
; CHECK-NEXT:    %[[T1:.*]] = load i32, ptr %[[b3]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.tail:
; CHECK-NEXT:    %[[v5:.*]] = phi i32 [ %[[T0]], %entry.select.true ],
; CHECK-SAME:                         [ %[[T1]], %entry.select.false ]
; CHECK-NEXT:    call void @consume(i32 %[[v5]])
; CHECK-NEXT:    call void @consume_ptr(ptr %[[a]])
; CHECK-NEXT:    call void @consume_ptr(ptr %[[b]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %a = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 0
  %b = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 1
  %p = select i1 %cond, ptr %a, ptr %b
  %v = load i32, ptr %p, align 4
  call void @consume(i32 %v)
  call void @consume_ptr(ptr %a)
  call void @consume_ptr(ptr %b)
  ret void
}

define void @select.noop.1(ptr %ctx, i1 %cond, i64 %idx) {
; CHECK:       define void @select.noop.1(ptr %[[ctx:.*]], i1 %[[cond:.*]], i64 %[[idx:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[a:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 %[[idx]], i32 0
; CHECK-NEXT:    %[[b:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 %[[idx]], i32 1
; CHECK-NEXT:    %[[p:.*]] = select i1 %[[cond]], ptr %[[a]], ptr %[[b]]
; CHECK-NEXT:    %[[v:.*]] = load i32, ptr %[[p]], align 4
; CHECK-NEXT:    call void @consume(i32 %[[v]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %a = getelementptr inbounds %struct.foo, ptr %ctx, i64 %idx, i32 0
  %b = getelementptr inbounds %struct.foo, ptr %ctx, i64 %idx, i32 1
  %p = select i1 %cond, ptr %a, ptr %b
  %v = load i32, ptr %p, align 4
  call void @consume(i32 %v)
  ret void
}

define void @select.pai.1(ptr %ctx, i1 %cond) {
; CHECK:       define void @select.pai.1(ptr %[[ctx:.*]], i1 %[[cond:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[pai_offset:.*]] = load i64, ptr @"llvm.bpf_sockopt:0:20$0:3", align 8
; CHECK-NEXT:    br i1 %[[cond]], label %entry.select.true, label %entry.select.false
; CHECK:       entry.select.true:
; CHECK-NEXT:    %[[ctx_offset2:.*]] = getelementptr i8, ptr %[[ctx]], i64 %[[pai_offset]]
; CHECK-NEXT:    %[[a3:.*]] = call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %[[ctx_offset2]])
; CHECK-NEXT:    %[[T0:.*]] = load i32, ptr %[[a3]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.false:
; CHECK-NEXT:    %[[b4:.*]] = getelementptr inbounds %struct.foo, ptr %[[ctx]], i64 0, i32 1
; CHECK-NEXT:    %[[T1:.*]] = load i32, ptr %[[b4]], align 4
; CHECK-NEXT:    br label %entry.select.tail
; CHECK:       entry.select.tail:
; CHECK-NEXT:    %[[v6:.*]] = phi i32 [ %[[T0]], %entry.select.true ],
; CHECK-SAME:                         [ %[[T1]], %entry.select.false ]
; CHECK-NEXT:    call void @consume(i32 %[[v6]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %pai.offset = load i64, ptr @"llvm.bpf_sockopt:0:20$0:3", align 8
  %ctx.offset = getelementptr i8, ptr %ctx, i64 %pai.offset
  %a = call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %ctx.offset)
  %b = getelementptr inbounds %struct.foo, ptr %ctx, i64 0, i32 1
  %p = select i1 %cond, ptr %a, ptr %b
  %v = load i32, ptr %p, align 4
  call void @consume(i32 %v)
  ret void
}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bpf_sockopt", file: !1, line: 9, size: 192, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{!3, !3, !3, !3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
