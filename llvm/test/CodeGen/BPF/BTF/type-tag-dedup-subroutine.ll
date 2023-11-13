; RUN: llc -march=bpfel -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
; RUN: llc -march=bpfeb -filetype=obj -o - %s \
; RUN: | llvm-objcopy --dump-section .BTF=- - | %python %S/print_btf.py - | FileCheck %s
;
; Source:
;
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   #define __tag3 __attribute__((btf_type_tag("tag3")))
;
;   struct bar {
;     void (__tag1 *aa)(int, int);
;     void (__tag2 *bb)(int, int);
;     void (*cc)(int, int);
;     int (__tag3 *dd)(int, int);
;   };
;
;   void root(struct bar *bar) {}
;
; Compilation flag:
;   clang -S -g -emit-llvm test.c -o test.ll

; CHECK:      [[[#]]] STRUCT 'bar' size=32 vlen=4
; CHECK-NEXT:   'aa' type_id=[[#ptag1:]] bits_offset=0
; CHECK-NEXT:   'bb' type_id=[[#ptag2:]] bits_offset=64
; CHECK-NEXT:   'cc' type_id=[[#pfunc:]] bits_offset=128
; CHECK-NEXT:   'dd' type_id=[[#ptag3:]] bits_offset=192
; CHECK-NEXT: [[[#ptag1]]] PTR '(anon)' type_id=[[#tag1:]]
; CHECK-NEXT: [[[#func:]]] FUNC_PROTO '(anon)' ret_type_id=0 vlen=2
; CHECK-NEXT:   '(anon)' type_id=[[#int:]]
; CHECK-NEXT:   '(anon)' type_id=[[#int]]
; CHECK-NEXT: [[[#tag1]]] TYPE_TAG 'tag1' type_id=[[#func]]
; CHECK-NEXT: [[[#int]]] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-NEXT: [[[#ptag2]]] PTR '(anon)' type_id=[[#tag2:]]
; CHECK-NEXT: [[[#tag2]]] TYPE_TAG 'tag2' type_id=[[#func]]
; CHECK-NEXT: [[[#pfunc]]] PTR '(anon)' type_id=[[#func]]
; CHECK-NEXT: [[[#ptag3]]] PTR '(anon)' type_id=[[#tag3:]]
; CHECK-NEXT: [[[#func2:]]] FUNC_PROTO '(anon)' ret_type_id=[[#int]] vlen=2
; CHECK-NEXT:   '(anon)' type_id=[[#int]]
; CHECK-NEXT:   '(anon)' type_id=[[#int]]
; CHECK-NEXT: [[[#tag3]]] TYPE_TAG 'tag1' type_id=[[#func2]]

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @root(ptr noundef %bar) #0 !dbg !10 {
entry:
  %bar.addr = alloca ptr, align 8
  store ptr %bar, ptr %bar.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %bar.addr, metadata !40, metadata !DIExpression()), !dbg !41
  ret void, !dbg !42
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git a2924ede5b9d936900b4d6bf00ee3567ffadd483)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test2.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "9b641c74e61a0c077fbe39d41f932190")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git a2924ede5b9d936900b4d6bf00ee3567ffadd483)"}
!10 = distinct !DISubprogram(name: "root", scope: !1, file: !1, line: 12, type: !11, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !1, line: 5, size: 256, elements: !15)
!15 = !{!16, !25, !32, !35}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "aa", scope: !14, file: !1, line: 6, baseType: !17, size: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64, annotations: !23)
!18 = !DISubroutineType(types: !19, annotations: !21)
!19 = !{null, !20, !20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{!22}
!22 = !{!"btf:type_tag", !"tag1"}
!23 = !{!24}
!24 = !{!"btf_type_tag", !"tag1"}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "bb", scope: !14, file: !1, line: 7, baseType: !26, size: 64, offset: 64)
!26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !27, size: 64, annotations: !30)
!27 = !DISubroutineType(types: !19, annotations: !28)
!28 = !{!29}
!29 = !{!"btf:type_tag", !"tag2"}
!30 = !{!31}
!31 = !{!"btf_type_tag", !"tag2"}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "cc", scope: !14, file: !1, line: 8, baseType: !33, size: 64, offset: 128)
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !34, size: 64)
!34 = !DISubroutineType(types: !19)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "dd", scope: !14, file: !1, line: 9, baseType: !36, size: 64, offset: 192)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64, annotations: !23)
!37 = !DISubroutineType(types: !38, annotations: !21)
!38 = !{!20, !20, !20}
!39 = !{}
!40 = !DILocalVariable(name: "bar", arg: 1, scope: !10, file: !1, line: 12, type: !13)
!41 = !DILocation(line: 12, column: 23, scope: !10)
!42 = !DILocation(line: 12, column: 29, scope: !10)
