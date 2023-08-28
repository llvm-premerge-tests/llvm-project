; RUN: opt -passes=bpf-context-marker -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((btf_decl_tag("ctx")))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct bar {
;      int a;
;      int b;
;    } __pai;
;    
;    struct buz {
;      int _1;
;      struct bar *b;
;    } __pai __ctx;
;    
;    void foo(struct buz *p) {
;      p->b->b = 42;
;    }
;    
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.buz = type { i32, ptr }
%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @foo(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !25
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.buz) %0, i32 1, i32 1), !dbg !25, !llvm.preserve.access.index !10
  %2 = load ptr, ptr %1, align 8, !dbg !25, !tbaa !26
  %3 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %2, i32 1, i32 1), !dbg !32, !llvm.preserve.access.index !16
  store i32 42, ptr %3, align 4, !dbg !33, !tbaa !34
  ret void, !dbg !36
}

; CHECK:      define dso_local void @foo(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   %[[v5:.*]] = call ptr (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.p0
; CHECK-SAME:       (ptr readonly elementtype(%struct.buz) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 3, i1 true, i32 immarg 0, i32 immarg 1)
; CHECK-SAME:      #[[v6:.*]], !tbaa
; CHECK-NEXT:   %[[v8:.*]] =
; CHECK-SAME:     call ptr @llvm.preserve.struct.access.index.p0.p0
; CHECK-SAME:       (ptr elementtype(%struct.bar) %[[v5]], i32 1, i32 1),
; CHECK-SAME:        !dbg ![[#]], !llvm.preserve.access.index ![[#]]
; CHECK-NEXT:   store i32 42, ptr %[[v8]], align 4, !dbg ![[#]], !tbaa
; CHECK-NEXT:   ret void, !dbg
; CHECK-NEXT: }

; CHECK     : attributes #[[v6]] = { memory(argmem: read) }


; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.context.marker.bpf(ptr readnone) #1

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
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !6, line: 14, type: !7, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!6 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !6, line: 9, size: 128, elements: !11, annotations: !20)
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !10, file: !6, line: 10, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !6, line: 11, baseType: !15, size: 64, offset: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !6, line: 4, size: 64, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !6, line: 5, baseType: !13, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !16, file: !6, line: 6, baseType: !13, size: 32, offset: 32)
!20 = !{!21}
!21 = !{!"btf_decl_tag", !"ctx"}
!22 = !{!23}
!23 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !6, line: 14, type: !9)
!24 = !DILocation(line: 0, scope: !5)
!25 = !DILocation(line: 15, column: 6, scope: !5)
!26 = !{!27, !31, i64 8}
!27 = !{!"buz", !28, i64 0, !31, i64 8}
!28 = !{!"int", !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C/C++ TBAA"}
!31 = !{!"any pointer", !29, i64 0}
!32 = !DILocation(line: 15, column: 9, scope: !5)
!33 = !DILocation(line: 15, column: 11, scope: !5)
!34 = !{!35, !28, i64 4}
!35 = !{!"bar", !28, i64 0, !28, i64 4}
!36 = !DILocation(line: 16, column: 1, scope: !5)
