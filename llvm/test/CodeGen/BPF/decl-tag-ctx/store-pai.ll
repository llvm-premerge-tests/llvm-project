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
;    struct buz {
;      int _1;
;      int _2;
;      int _3;
;      union {
;        struct foo a;
;        struct bar b[7];
;      };
;    } __pai __ctx;
;    
;    void quux(struct buz *p) {
;      p->b[5].b = 42;
;    }
;    
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes \
;         -debug-info-kind=limited -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.buz = type { i32, i32, i32, %union.anon }
%union.anon = type { [7 x %struct.bar] }
%struct.bar = type { i32, i32 }

; Function Attrs: nounwind
define dso_local void @quux(ptr noundef %p) #0 !dbg !34 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !39, metadata !DIExpression()), !dbg !40
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !41
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.buz) %0, i32 3, i32 3), !dbg !41, !llvm.preserve.access.index !5
  %2 = call ptr @llvm.preserve.union.access.index.p0.p0(ptr %1, i32 1), !dbg !41, !llvm.preserve.access.index !3
  %3 = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([7 x %struct.bar]) %2, i32 1, i32 5), !dbg !42, !llvm.preserve.access.index !24
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bar) %3, i32 1, i32 1), !dbg !43, !llvm.preserve.access.index !25
  store i32 42, ptr %4, align 4, !dbg !44, !tbaa !45
  ret void, !dbg !48
}

; CHECK:      define dso_local void @quux(ptr noundef %[[p:.*]]) {{.*}} {
; CHECK-NEXT: entry:
; CHECK-NEXT:   call void @llvm.dbg.value
; CHECK-NEXT:   call void (i32, ptr, i1, i8, i8, i8, i1, ...)
; CHECK-SAME:     @llvm.bpf.getelementptr.and.store.i32
; CHECK-SAME:       (i32 42,
; CHECK-SAME:        ptr writeonly elementtype(i8) %[[p]],
; CHECK-SAME:        i1 false, i8 0, i8 1, i8 2, i1 true, i64 immarg 56)
; CHECK-SAME:      #[[v5:.*]], !tbaa
; CHECK-NEXT:   ret void, !dbg
; CHECK-NEXT: }

; CHECK:      attributes #[[v5]] = { memory(argmem: write) }

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.context.marker.bpf(ptr readnone) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.union.access.index.p0.p0(ptr, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31, !32}
!llvm.ident = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{!3, !24}
!3 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !5, file: !4, line: 17, size: 448, elements: !14)
!4 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "buz", file: !4, line: 13, size: 544, elements: !6, annotations: !12)
!6 = !{!7, !9, !10, !11}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "_1", scope: !5, file: !4, line: 14, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "_2", scope: !5, file: !4, line: 15, baseType: !8, size: 32, offset: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "_3", scope: !5, file: !4, line: 16, baseType: !8, size: 32, offset: 64)
!11 = !DIDerivedType(tag: DW_TAG_member, scope: !5, file: !4, line: 17, baseType: !3, size: 448, offset: 96)
!12 = !{!13}
!13 = !{!"btf_decl_tag", !"ctx"}
!14 = !{!15, !23}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !4, line: 18, baseType: !16, size: 80)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !4, line: 4, size: 80, elements: !17)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !4, line: 5, baseType: !19, size: 80)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 80, elements: !21)
!20 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!21 = !{!22}
!22 = !DISubrange(count: 10)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !4, line: 19, baseType: !24, size: 448)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !25, size: 448, elements: !29)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !4, line: 8, size: 64, elements: !26)
!26 = !{!27, !28}
!27 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !25, file: !4, line: 9, baseType: !8, size: 32)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !25, file: !4, line: 10, baseType: !8, size: 32, offset: 32)
!29 = !{!30}
!30 = !DISubrange(count: 7)
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 1, !"wchar_size", i32 4}
!33 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 52515717a1cb9f6dda0c91bce7b9ea4f7563b9f8)"}
!34 = distinct !DISubprogram(name: "quux", scope: !4, file: !4, line: 23, type: !35, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !38)
!35 = !DISubroutineType(types: !36)
!36 = !{null, !37}
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!38 = !{!39}
!39 = !DILocalVariable(name: "p", arg: 1, scope: !34, file: !4, line: 23, type: !37)
!40 = !DILocation(line: 0, scope: !34)
!41 = !DILocation(line: 24, column: 6, scope: !34)
!42 = !DILocation(line: 24, column: 3, scope: !34)
!43 = !DILocation(line: 24, column: 11, scope: !34)
!44 = !DILocation(line: 24, column: 13, scope: !34)
!45 = !{!46, !46, i64 0}
!46 = !{!"omnipotent char", !47, i64 0}
!47 = !{!"Simple C/C++ TBAA"}
!48 = !DILocation(line: 25, column: 1, scope: !34)
