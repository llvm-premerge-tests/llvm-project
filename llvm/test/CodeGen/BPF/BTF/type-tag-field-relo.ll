; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck %s
;
; Source:
;
;   #define __pai  __attribute__((preserve_access_index));
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;
;   struct alpha {
;     int zulu;
;   } __pai;
;
;   struct bravo {
;     struct alpha __tag1 *yankee;
;   } __pai;
;
;   int func(struct bravo *xray) {
;     return xray->yankee->zulu;
;   }
;
; Compilation command:
; 
;   cat test.c | clang -x c -target bpf -O2 -g -emit-llvm -S - -o -
;
; The relocation entry for zulu should point to STRUCT 'alpha',
; not TYPE_TAG 'tag1' -> STRUCT 'alpha'.

@"llvm.alpha:0:0$0:0" = external global i64, !llvm.preserve.access.index !0 #0
@"llvm.bravo:0:0$0:0" = external global i64, !llvm.preserve.access.index !8 #0

; Function Attrs: nofree nosync nounwind memory(read, inaccessiblemem: none)
define dso_local i32 @func(ptr noundef readonly %xray) local_unnamed_addr #1 !dbg !22 {
entry:
  call void @llvm.dbg.value(metadata ptr %xray, metadata !27, metadata !DIExpression()), !dbg !28
  %0 = load i64, ptr @"llvm.bravo:0:0$0:0", align 8
  %1 = getelementptr i8, ptr %xray, i64 %0
  %2 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 1, ptr %1)
  %3 = load ptr, ptr %2, align 8, !dbg !29, !tbaa !30
  %4 = load i64, ptr @"llvm.alpha:0:0$0:0", align 8
  %5 = getelementptr i8, ptr %3, i64 %4
  %6 = tail call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %5)
  %7 = load i32, ptr %6, align 4, !dbg !35, !tbaa !36
  ret i32 %7, !dbg !39
}

; Function Attrs: nofree nosync nounwind memory(none)
declare ptr @llvm.bpf.passthrough.p0.p0(i32, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { "btf_ama" }
attributes #1 = { nofree nosync nounwind memory(read, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!14}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DICompositeType(tag: DW_TAG_structure_type, name: "alpha", file: !1, line: 4, size: 32, elements: !2, annotations: !6)
!1 = !DIFile(filename: "<stdin>", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "89810ba04b039111ea709cb54ed653bc")
!2 = !{!3}
!3 = !DIDerivedType(tag: DW_TAG_member, name: "zulu", scope: !4, file: !1, line: 5, baseType: !5, size: 32)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "alpha", file: !1, line: 4, size: 32, elements: !2)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !{!"btf:type_tag", !"tag1"}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bravo", file: !1, line: 8, size: 64, elements: !9)
!9 = !{!10}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "yankee", scope: !8, file: !1, line: 9, baseType: !11, size: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !0, size: 64, annotations: !12)
!12 = !{!13}
!13 = !{!"btf_type_tag", !"tag1"}
!14 = distinct !DICompileUnit(language: DW_LANG_C11, file: !15, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 034dbeecee00be5c47fc265333c6d58bb4801240)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !16, splitDebugInlining: false, nameTableKind: None)
!15 = !DIFile(filename: "-", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "89810ba04b039111ea709cb54ed653bc")
!16 = !{!0}
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 034dbeecee00be5c47fc265333c6d58bb4801240)"}
!22 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 12, type: !23, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !14, retainedNodes: !26)
!23 = !DISubroutineType(types: !24)
!24 = !{!5, !25}
!25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!26 = !{!27}
!27 = !DILocalVariable(name: "xray", arg: 1, scope: !22, file: !1, line: 12, type: !25)
!28 = !DILocation(line: 0, scope: !22)
!29 = !DILocation(line: 13, column: 16, scope: !22)
!30 = !{!31, !32, i64 0}
!31 = !{!"bravo", !32, i64 0}
!32 = !{!"any pointer", !33, i64 0}
!33 = !{!"omnipotent char", !34, i64 0}
!34 = !{!"Simple C/C++ TBAA"}
!35 = !DILocation(line: 13, column: 24, scope: !22)
!36 = !{!37, !38, i64 0}
!37 = !{!"alpha", !38, i64 0}
!38 = !{!"int", !33, i64 0}
!39 = !DILocation(line: 13, column: 3, scope: !22)

; CHECK: [[L1:.Ltmp[0-9]+]]:
; CHECK:        r1 = *(u64 *)(r1 + 0)
; CHECK:        r0 = *(u32 *)(r1 + 0)

; CHECK:        .long   1                               # BTF_KIND_STRUCT(id = 2)
; CHECK:        .long   66                              # BTF_KIND_STRUCT(id = 7)
; CHECK:        .long   77                              # BTF_KIND_TYPE_TAG(id = 8)
;
; CHECK:        .ascii  "bravo"                         # string offset=1
; CHECK:        .ascii  "0:0"                           # string offset=34
; CHECK:        .ascii  "alpha"                         # string offset=66

; CHECK:        .long   16                              # FieldReloc
; CHECK-NEXT:   .long   28                              # Field reloc section string offset=28
; CHECK-NEXT:   .long   2
;
; Step #1: xray->yankee
; CHECK-NEXT:   .long   [[L1]]
; CHECK-NEXT:   .long   2
;                      ^^^ id 2 -> STRUCT 'bravo'
; CHECK-NEXT:   .long   34
;                      ^^^^ relocation spec '0:0'
; CHECK-NEXT:   .long   0
;
; Step #2: yankee->zulu
; CHECK-NEXT:   .long   .Ltmp[[#]]
; CHECK-NEXT:   .long   7
;                      ^^^ id 7 -> STRUCT 'alpha'
;                          The gist of the test-case is to check that
;                          it is not id 8, corresponding to TYPE_TAG.
; CHECK-NEXT:   .long   34
;                      ^^^^ relocation spec '0:0'
; CHECK-NEXT:   .long   0
