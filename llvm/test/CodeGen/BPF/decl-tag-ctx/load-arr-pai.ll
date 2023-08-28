; RUN: opt -passes=bpf-context-marker -mtriple=bpf-pc-linux -S -o - %s | FileCheck %s
;
; Source:
;    #define __ctx __attribute__((btf_decl_tag("ctx")))
;    #define __pai __attribute__((preserve_access_index))
;    
;    struct bar {
;      int a[7];
;    } __pai __ctx;
;    
;    int buz(struct bar *p) {
;      return p->a[5];
;    }
;    
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bar = type { [7 x i32] }

; Function Attrs: nounwind
define dso_local i32 @buz(ptr noundef %p) #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !21, metadata !DIExpression()), !dbg !22
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !23
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %0, i32 0, i32 0), !dbg !23, !llvm.preserve.access.index !15
  %2 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([7 x i32]) %1, i32 1, i32 5), !dbg !24, !llvm.preserve.access.index !3
  %3 = load i32, ptr %2, align 4, !dbg !24, !tbaa !25
  ret i32 %3, !dbg !29
}

; CHECK:      define dso_local i32 @buz(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   %[[v5:.*]] = call i32 (ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.load.i32
; CHECK-SAME:       (ptr readonly elementtype(%struct.bar) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i32 immarg 0, i32 immarg 0, i32 immarg 5)
; CHECK-SAME:      #[[v6:.*]], !tbaa
; CHECK-NEXT:   ret i32 %[[v5]]
; CHECK-NEXT: }

; CHECK:      attributes #[[v6]] = { memory(argmem: read) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.context.marker.bpf(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 224, elements: !5)
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!6}
!6 = !DISubrange(count: 7)
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)"}
!10 = distinct !DISubprogram(name: "buz", scope: !11, file: !11, line: 8, type: !12, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !20)
!11 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!12 = !DISubroutineType(types: !13)
!13 = !{!4, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !11, line: 4, size: 224, elements: !16, annotations: !18)
!16 = !{!17}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !15, file: !11, line: 5, baseType: !3, size: 224)
!18 = !{!19}
!19 = !{!"btf_decl_tag", !"ctx"}
!20 = !{!21}
!21 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !11, line: 8, type: !14)
!22 = !DILocation(line: 0, scope: !10)
!23 = !DILocation(line: 9, column: 13, scope: !10)
!24 = !DILocation(line: 9, column: 10, scope: !10)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocation(line: 9, column: 3, scope: !10)
