; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=obj < %s \
; RUN:   | llvm-objdump --no-show-raw-insn --no-addresses -d - \
; RUN:   | FileCheck %s

; Check that BPFMIPeephole::eliminateZExtSeq() knows how to handle
; "small" (32/16/8-bit) loads from CORE instructions.
;
; Generated from the following C code:
;   struct t {
;     unsigned char a;
;   } __attribute__((preserve_access_index));
;
;   unsigned int foo(struct t *t, unsigned long b, unsigned long *p) {
;     unsigned int a;
;     if (b)
;       a = t->a;
;     else
;       a = 0;
;     *p = a;
;     return a;
;   }
;
; Using the following command:
;   clang -g -O2 -emit-llvm -S --target=bpf t.c -o t.ll

@"llvm.t:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0

; Function Attrs: nofree nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none)
define dso_local i32 @foo(ptr noundef readonly %t, i64 noundef %b, ptr nocapture noundef writeonly %p) local_unnamed_addr #1 !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata ptr %t, metadata !20, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata i64 %b, metadata !21, metadata !DIExpression()), !dbg !24
  call void @llvm.dbg.value(metadata ptr %p, metadata !22, metadata !DIExpression()), !dbg !24
  %tobool.not = icmp eq i64 %b, 0, !dbg !25
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !27

if.then:                                          ; preds = %entry
  %0 = load i64, ptr @"llvm.t:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %t, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %1)
  %3 = load i8, ptr %2, align 1, !dbg !28, !tbaa !29
  %conv = zext i8 %3 to i32, !dbg !33
  call void @llvm.dbg.value(metadata i32 %conv, metadata !23, metadata !DIExpression()), !dbg !24
  br label %if.end, !dbg !34

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi i32 [ %conv, %if.then ], [ 0, %entry ], !dbg !35
  call void @llvm.dbg.value(metadata i32 %a.0, metadata !23, metadata !DIExpression()), !dbg !24
  %conv1 = zext i32 %a.0 to i64, !dbg !36
  store i64 %conv1, ptr %p, align 8, !dbg !37, !tbaa !38
  ret i32 %a.0, !dbg !40
}

; CHECK:      <foo>:
; CHECK-NEXT:        w0 = 0x0
; CHECK-NEXT:        if r2 == 0x0 goto +0x1 <[[L:.*]]>
; CHECK-NEXT:        r0 = *(u8 *)(r1 + 0x0)
; CHECK-EMPTY:
; CHECK-NEXT: <[[L]]>:
; CHECK-NEXT:        *(u64 *)(r3 + 0x0) = r0
; CHECK-NEXT:        exit

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: !1, line: 1, size: 8, elements: !2)
!1 = !DIFile(filename: "t.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "6232d59853f85f13ad6bb49cfe4de63d")
!2 = !{!3}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !0, file: !1, line: 2, baseType: !4, size: 8)
!4 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!5 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang cf42dc00d29d1b1cc97262051fef95237e9c2fe3)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!11 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang cf42dc00d29d1b1cc97262051fef95237e9c2fe3)"}
!12 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !13, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !19)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !16, !17, !18}
!15 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !0, size: 64)
!17 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!19 = !{!20, !21, !22, !23}
!20 = !DILocalVariable(name: "t", arg: 1, scope: !12, file: !1, line: 5, type: !16)
!21 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !1, line: 5, type: !17)
!22 = !DILocalVariable(name: "p", arg: 3, scope: !12, file: !1, line: 5, type: !18)
!23 = !DILocalVariable(name: "a", scope: !12, file: !1, line: 6, type: !15)
!24 = !DILocation(line: 0, scope: !12)
!25 = !DILocation(line: 7, column: 7, scope: !26)
!26 = distinct !DILexicalBlock(scope: !12, file: !1, line: 7, column: 7)
!27 = !DILocation(line: 7, column: 7, scope: !12)
!28 = !DILocation(line: 8, column: 12, scope: !26)
!29 = !{!30, !31, i64 0}
!30 = !{!"t", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 8, column: 9, scope: !26)
!34 = !DILocation(line: 8, column: 5, scope: !26)
!35 = !DILocation(line: 0, scope: !26)
!36 = !DILocation(line: 11, column: 8, scope: !12)
!37 = !DILocation(line: 11, column: 6, scope: !12)
!38 = !{!39, !39, i64 0}
!39 = !{!"long", !31, i64 0}
!40 = !DILocation(line: 12, column: 3, scope: !12)
