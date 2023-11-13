; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   int __tag1 * __tag1 __tag2 *g;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

@g = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17, !18}
!llvm.ident = !{!19}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 5aa6dd1e09616a455377f3066d2034d3e8a073ba)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "9ae91fe3dd8e44985841816e35923786")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, annotations: !13)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, annotations: !10)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !8)
!8 = !{!9}
!9 = !{!"btf:type_tag", !"tag1"}
!10 = !{!9, !11, !12}
!11 = !{!"btf:type_tag", !"tag2"}
!12 = !{!"btf_type_tag", !"tag1"}
!13 = !{!12, !14}
!14 = !{!"btf_type_tag", !"tag2"}

; CHECK:             .long   0                               # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   0                               # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   6
; CHECK-NEXT:        .long   1                               # BTF_KIND_TYPE_TAG(id = 3)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   6                               # BTF_KIND_TYPE_TAG(id = 4)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   11                              # BTF_KIND_INT(id = 5)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020
; CHECK-NEXT:        .long   1                               # BTF_KIND_TYPE_TAG(id = 6)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   5
; CHECK-NEXT:        .long   15                              # BTF_KIND_VAR(id = 7)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   17                              # BTF_KIND_DATASEC(id = 8)
; CHECK-NEXT:        .long   251658241                       # 0xf000001
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   g
; CHECK-NEXT:        .long   8

; CHECK:             .ascii  "tag1"                          # string offset=1
; CHECK:             .ascii  "tag2"                          # string offset=6
; CHECK:             .ascii  "int"                           # string offset=11
; CHECK:             .byte   103                             # string offset=15

!15 = !{i32 7, !"Dwarf Version", i32 5}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 4}
!18 = !{i32 7, !"frame-pointer", i32 2}
!19 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 5aa6dd1e09616a455377f3066d2034d3e8a073ba)"}
