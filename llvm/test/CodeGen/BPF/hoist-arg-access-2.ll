; RUN: opt -S -passes=bpf-hoist-argument-access -mtriple=bpf-pc-linux < %s | FileCheck %s

; Input generated from the following C code:
;
;   #define __pai __attribute__((preserve_access_index))
;
;   struct bpf_sock {
;     int bound_dev_if;
;     int family;
;   } __ctx;
;
;   struct bpf_sockopt {
;     int _;
;     struct bpf_sock *sk;
;     int level;
;     int optlen;
;   } __pai;
;
;   extern void magic2(int);
;
;   void known_load_sink_example_1(struct bpf_sockopt *ctx)
;   {
;     unsigned g = 0;
;     if (ctx->level == 42)
;       magic2(ctx->sk->family);
;     else
;       magic2(ctx->optlen);
;   }
;
; Using command:
;
;   clang --target=bpf -emit-llvm -g -O2 -S -o -
;
; Plus some manual cleanup of unused metadata.

%struct.bpf_sock = type { i32, i32 }

@"llvm.bpf_sockopt:0:16$0:2" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.bpf_sockopt:0:20$0:3" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.bpf_sockopt:0:8$0:1" = external global i64, !llvm.preserve.access.index !0 #0

define void @known_load_sink_example_1(ptr noundef readonly %ctx) {
; CHECK:       define void @known_load_sink_example_1(ptr noundef readonly %[[ctx:.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %[[T0:.*]] = load i64, ptr @"llvm.bpf_sockopt:0:16$0:2", align 8
; CHECK-NEXT:    %[[T1:.*]] = getelementptr i8, ptr %[[ctx]], i64 %[[T0]]
; CHECK-NEXT:    %[[T2:.*]] = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %[[T1]])
; CHECK-NEXT:    %[[T3:.*]] = load i32, ptr %[[T2]], align 8
; CHECK-NEXT:    %[[cmp:.*]] = icmp eq i32 %[[T3]], 42
; CHECK-NEXT:    br i1 %[[cmp]], label %if.then, label %if.else
; CHECK:       if.then:
; CHECK-NEXT:    %[[T4:.*]] = load i64, ptr @"llvm.bpf_sockopt:0:8$0:1", align 8
; CHECK-NEXT:    %[[T5:.*]] = getelementptr i8, ptr %[[ctx]], i64 %[[T4]]
; CHECK-NEXT:    %[[T6:.*]] = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %[[T5]])
; CHECK-NEXT:    %[[T7:.*]] = load ptr, ptr %[[T6]], align 8
; CHECK-NEXT:    %[[family:.*]] = getelementptr inbounds %struct.bpf_sock, ptr %[[T7]], i64 0, i32 1
; CHECK-NEXT:    %[[T8:.*]] = load i32, ptr %[[family]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.else:
; CHECK-NEXT:    %[[T9:.*]] = load i64, ptr @"llvm.bpf_sockopt:0:20$0:3", align 8
; CHECK-NEXT:    %[[T10:.*]] = getelementptr i8, ptr %[[ctx]], i64 %[[T9]]
; CHECK-NEXT:    %[[T11:.*]] = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %[[T10]])
; CHECK-NEXT:    %[[T12:.*]] = load i32, ptr %[[T11]], align 4
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    %[[T13:.*]] = phi i32 [ %[[T12]], %if.else ],
; CHECK-SAME:                          [ %[[T8]], %if.then ]
; CHECK-NEXT:    tail call void @magic2(i32 noundef %[[T13]])
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %0 = load i64, ptr @"llvm.bpf_sockopt:0:16$0:2", align 8
  %1 = getelementptr i8, ptr %ctx, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  %3 = load i32, ptr %2, align 8
  %cmp = icmp eq i32 %3, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %4 = load i64, ptr @"llvm.bpf_sockopt:0:8$0:1", align 8
  %5 = getelementptr i8, ptr %ctx, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 2, ptr %5)
  %7 = load ptr, ptr %6, align 8
  %family = getelementptr inbounds %struct.bpf_sock, ptr %7, i64 0, i32 1
  br label %if.end

if.else:                                          ; preds = %entry
  %8 = load i64, ptr @"llvm.bpf_sockopt:0:20$0:3", align 8
  %9 = getelementptr i8, ptr %ctx, i64 %8
  %10 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %9)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %.sink3 = phi ptr [ %10, %if.else ], [ %family, %if.then ]
  %11 = load i32, ptr %.sink3, align 4
  tail call void @magic2(i32 noundef %11)
  ret void
}

declare void @magic2(i32 noundef)

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr)

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bpf_sockopt", file: !1, line: 9, size: 192, elements: !2)
!1 = !DIFile(filename: "some.file", directory: "/some/dir", checksumkind: CSK_MD5, checksum: "00000000000000000000000000000000")
!2 = !{!3, !5, !13, !14}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "_", scope: !0, file: !1, line: 10, baseType: !4, size: 32)
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIDerivedType(tag: DW_TAG_member, name: "sk", scope: !0, file: !1, line: 11, baseType: !6, size: 64, offset: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bpf_sock", file: !1, line: 4, size: 64, elements: !8)
!8 = !{!9, !10}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "bound_dev_if", scope: !7, file: !1, line: 5, baseType: !4, size: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "family", scope: !7, file: !1, line: 6, baseType: !4, size: 32, offset: 32)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "level", scope: !0, file: !1, line: 12, baseType: !4, size: 32, offset: 128)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "optlen", scope: !0, file: !1, line: 13, baseType: !4, size: 32, offset: 160)
