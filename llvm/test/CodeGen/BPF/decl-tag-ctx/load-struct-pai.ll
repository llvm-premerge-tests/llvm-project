; RUN: opt -passes=bpf-context-marker -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((btf_decl_tag("ctx")))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct foo {
;      int a;
;      int b;
;    };
;    
;    struct bar {
;      int _1;
;      int _2;
;      struct foo c;
;    } __pai __ctx;
;    
;    int buz(struct bar *p) {
;      return p->c.b;
;    }
;    
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { i32, i32, %struct.foo }
%struct.foo = type { i32, i32 }

; Function Attrs: nounwind
define dso_local i32 @buz(ptr noundef %p) #0 !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !25
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %0, i32 2, i32 2), !dbg !25, !llvm.preserve.access.index !11
  %b = getelementptr inbounds %struct.foo, ptr %1, i32 0, i32 1, !dbg !26
  %2 = load i32, ptr %b, align 4, !dbg !26, !tbaa !27
  ret i32 %2, !dbg !33
}

; CHECK:      define dso_local i32 @buz(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   %[[b1:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(%struct.bar) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 2, i32 immarg 1)
; CHECK-SAME:      #[[v5:.*]], !tbaa
; CHECK-NEXT:   ret i32 %[[b1]]
; CHECK-NEXT: }

; CHECK:      attributes #[[v5]] = { memory(argmem: read) }


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
!5 = distinct !DISubprogram(name: "buz", scope: !6, file: !6, line: 15, type: !7, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !22)
!6 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !6, line: 9, size: 128, elements: !12, annotations: !20)
!12 = !{!13, !14, !15}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !11, file: !6, line: 10, baseType: !9, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "_2", scope: !11, file: !6, line: 11, baseType: !9, size: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !11, file: !6, line: 12, baseType: !16, size: 64, offset: 64)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !6, line: 4, size: 64, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !6, line: 5, baseType: !9, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !16, file: !6, line: 6, baseType: !9, size: 32, offset: 32)
!20 = !{!21}
!21 = !{!"btf_decl_tag", !"ctx"}
!22 = !{!23}
!23 = !DILocalVariable(name: "p", arg: 1, scope: !5, file: !6, line: 15, type: !10)
!24 = !DILocation(line: 0, scope: !5)
!25 = !DILocation(line: 16, column: 13, scope: !5)
!26 = !DILocation(line: 16, column: 15, scope: !5)
!27 = !{!28, !29, i64 12}
!28 = !{!"bar", !29, i64 0, !29, i64 4, !32, i64 8}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !{!"foo", !29, i64 0, !29, i64 4}
!33 = !DILocation(line: 16, column: 3, scope: !5)
