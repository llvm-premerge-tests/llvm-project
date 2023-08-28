; RUN: opt -passes=bpf-context-marker -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((btf_decl_tag("ctx")))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct foo {
;      char a[10];
;    } __pai;
;    
;    struct bar {
;      int a;
;      int b;
;    } __pai;
;    
;    union buz {
;      struct foo a;
;      struct bar b;
;    } __pai __ctx;
;    
;    int quux(union buz *p) {
;      return p->b.b;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @quux(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !29, metadata !DIExpression()), !dbg !30
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !31
  %1 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %0, i32 1), !dbg !31, !llvm.preserve.access.index !11
  %2 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %1, i32 1, i32 1), !dbg !32, !llvm.preserve.access.index !22
  %3 = load i32, ptr %2, align 4, !dbg !32, !tbaa !33
  ret i32 %3, !dbg !36
}

; CHECK:      define dso_local i32 @quux(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   %[[v5:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(%struct.bar) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 1)
; CHECK-SAME:      #[[v6:.*]], !tbaa
; CHECK-NEXT:   ret i32 %[[v5]]
; CHECK-NEXT: }

; CHECK:      attributes #[[v6]] = { memory(argmem: read) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.context.marker.bpf(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)"}
!5 = distinct !DISubprogram(name: "quux", scope: !6, file: !6, line: 18, type: !7, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !28)
!6 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "buz", file: !6, line: 13, size: 96, elements: !12, annotations: !26)
!12 = !{!13, !21}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !6, line: 14, baseType: !14, size: 80)
!14 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !6, line: 4, size: 80, elements: !15)
!15 = !{!16}
!16 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !14, file: !6, line: 5, baseType: !17, size: 80)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 80, elements: !19)
!18 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!19 = !{!20}
!20 = !DISubrange(count: 10)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !6, line: 15, baseType: !22, size: 64)
!22 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !6, line: 8, size: 64, elements: !23)
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !22, file: !6, line: 9, baseType: !9, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !22, file: !6, line: 10, baseType: !9, size: 32, offset: 32)
!26 = !{!27}
!27 = !{!"btf_decl_tag", !"ctx"}
!28 = !{!29}
!29 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !6, line: 18, type: !10)
!30 = !DILocation(line: 0, scope: !5)
!31 = !DILocation(line: 19, column: 13, scope: !5)
!32 = !DILocation(line: 19, column: 15, scope: !5)
!33 = !{!34, !34, i64 0}
!34 = !{!"omnipotent char", !35, i64 0}
!35 = !{!"Simple C/C++ TBAA"}
!36 = !DILocation(line: 19, column: 3, scope: !5)
