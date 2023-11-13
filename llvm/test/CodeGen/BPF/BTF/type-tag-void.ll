; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   void __tag1 *foo;
; Compilation flag:
;   clang -S -g -emit-llvm test.c -o test.ll

@foo = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git b0d4d11b535f4c3b730222013f4da5b0cbc4558a)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "d51e2505a6f475a2c1811fd5e5fa9c49")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, annotations: !9)
!6 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void", annotations: !7)
!7 = !{!8}
!8 = !{!"btf:type_tag", !"tag1"}
!9 = !{!10}
!10 = !{!"btf_type_tag", !"tag1"}
!11 = !{i32 7, !"Dwarf Version", i32 5}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 7, !"uwtable", i32 2}
!17 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git b0d4d11b535f4c3b730222013f4da5b0cbc4558a)"}

; CHECK:             .long   0                               # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                        # 0x2000000
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   1                               # BTF_KIND_TYPE_TAG(id = 2)
; CHECK-NEXT:        .long   301989888                       # 0x12000000
; CHECK-NEXT:        .long   0
;                           ^^^ void type id
; CHECK-NEXT:        .long   6                               # BTF_KIND_VAR(id = 3)
; CHECK-NEXT:        .long   234881024                       # 0xe000000
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   1

; CHECK:             .ascii  "tag1"                          # string offset=1
; CHECK:             .ascii  "foo"                           # string offset=6
