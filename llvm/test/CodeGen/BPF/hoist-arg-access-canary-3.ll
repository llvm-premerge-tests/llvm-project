; RUN: opt -O2 -mtriple=bpf-pc-linux -mcpu=v4 -o - %s \
; RUN:   | llc -mcpu=v4 - | FileCheck %s
;
; Source:
;    struct bpf_sockopt {
;      int family;
;      int level;
;      int optlen;
;    } __attribute__((preserve_access_index));
;    
;    extern void consume(int);
;    
;    void load_sink_example(struct bpf_sockopt *ctx)
;    {
;      if (ctx->level == 42)
;        consume(ctx->family);
;      else
;        consume(ctx->optlen);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -debug-info-kind=limited

%struct.bpf_sockopt = type { i32, i32, i32 }

; Function Attrs: nounwind
define dso_local void @load_sink_example(ptr noundef %ctx) #0 !dbg !5 {
; CHECK: {{.*}} = *(u32 *)(r1 + 4)
; CHECK: w1 = *(u32 *)(r1 + 0)
; CHECK: w1 = *(u32 *)(r1 + 8)
entry:
  %ctx.addr = alloca ptr, align 8
  store ptr %ctx, ptr %ctx.addr, align 8, !tbaa !18
  call void @llvm.dbg.declare(metadata ptr %ctx.addr, metadata !17, metadata !DIExpression()), !dbg !22
  %0 = load ptr, ptr %ctx.addr, align 8, !dbg !23, !tbaa !18
  %1 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bpf_sockopt) %0, i32 1, i32 1), !dbg !25, !llvm.preserve.access.index !10
  %2 = load i32, ptr %1, align 4, !dbg !25, !tbaa !26
  %cmp = icmp eq i32 %2, 42, !dbg !29
  br i1 %cmp, label %if.then, label %if.else, !dbg !30

if.then:                                          ; preds = %entry
  %3 = load ptr, ptr %ctx.addr, align 8, !dbg !31, !tbaa !18
  %4 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bpf_sockopt) %3, i32 0, i32 0), !dbg !32, !llvm.preserve.access.index !10
  %5 = load i32, ptr %4, align 4, !dbg !32, !tbaa !33
  call void @consume(i32 noundef %5), !dbg !34
  br label %if.end, !dbg !34

if.else:                                          ; preds = %entry
  %6 = load ptr, ptr %ctx.addr, align 8, !dbg !35, !tbaa !18
  %7 = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.bpf_sockopt) %6, i32 2, i32 2), !dbg !36, !llvm.preserve.access.index !10
  %8 = load i32, ptr %7, align 4, !dbg !36, !tbaa !37
  call void @consume(i32 noundef %8), !dbg !38
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !39
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg) #2

declare void @consume(i32 noundef) #3

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0 (/home/eddy/work/llvm-project/clang 53c495f835926142c10c80d7d0505f59b1e46e49)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 53c495f835926142c10c80d7d0505f59b1e46e49)"}
!5 = distinct !DISubprogram(name: "load_sink_example", scope: !6, file: !6, line: 9, type: !7, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!6 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bpf_sockopt", file: !6, line: 1, size: 96, elements: !11)
!11 = !{!12, !14, !15}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "family", scope: !10, file: !6, line: 2, baseType: !13, size: 32)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "level", scope: !10, file: !6, line: 3, baseType: !13, size: 32, offset: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "optlen", scope: !10, file: !6, line: 4, baseType: !13, size: 32, offset: 64)
!16 = !{!17}
!17 = !DILocalVariable(name: "ctx", arg: 1, scope: !5, file: !6, line: 9, type: !9)
!18 = !{!19, !19, i64 0}
!19 = !{!"any pointer", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 9, column: 44, scope: !5)
!23 = !DILocation(line: 11, column: 7, scope: !24)
!24 = distinct !DILexicalBlock(scope: !5, file: !6, line: 11, column: 7)
!25 = !DILocation(line: 11, column: 12, scope: !24)
!26 = !{!27, !28, i64 4}
!27 = !{!"bpf_sockopt", !28, i64 0, !28, i64 4, !28, i64 8}
!28 = !{!"int", !20, i64 0}
!29 = !DILocation(line: 11, column: 18, scope: !24)
!30 = !DILocation(line: 11, column: 7, scope: !5)
!31 = !DILocation(line: 12, column: 13, scope: !24)
!32 = !DILocation(line: 12, column: 18, scope: !24)
!33 = !{!27, !28, i64 0}
!34 = !DILocation(line: 12, column: 5, scope: !24)
!35 = !DILocation(line: 14, column: 13, scope: !24)
!36 = !DILocation(line: 14, column: 18, scope: !24)
!37 = !{!27, !28, i64 8}
!38 = !DILocation(line: 14, column: 5, scope: !24)
!39 = !DILocation(line: 15, column: 1, scope: !5)

