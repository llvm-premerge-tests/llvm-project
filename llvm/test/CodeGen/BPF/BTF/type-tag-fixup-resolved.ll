; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;   struct foo {
;           int i;
;   };
;   struct map_value {
;           struct foo __tag2 __tag1 *ptr;
;   };
;   void func(struct map_value *, struct foo *);
;   void test(void)
;   {
;           struct map_value v = {};
;           func(&v, v.ptr);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.map_value = type { ptr }

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %v = alloca %struct.map_value, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %v) #4, !dbg !27
  call void @llvm.dbg.declare(metadata ptr %v, metadata !11, metadata !DIExpression()), !dbg !28
  store i64 0, ptr %v, align 8, !dbg !28
  call void @func(ptr noundef nonnull %v, ptr noundef null) #4, !dbg !29
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %v) #4, !dbg !30
  ret void, !dbg !30
}

; CHECK:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:   .long   218103808                       # 0xd000000
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:   .long   201326593                       # 0xc000001
; CHECK-NEXT:   .long   1
; CHECK-NEXT:   .long   0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT:   .long   218103810                       # 0xd000002
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   7
; CHECK-NEXT:   .long   0                               # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:   .long   33554432                        # 0x2000000
; CHECK-NEXT:   .long   5
; CHECK-NEXT:   .long   35                              # BTF_KIND_STRUCT(id = 5)
; CHECK-NEXT:   .long   67108865                        # 0x4000001
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   45
; CHECK-NEXT:   .long   6
; CHECK-NEXT:   .long   0                               # 0x0
; CHECK-NEXT:   .long   0                               # BTF_KIND_PTR(id = 6)
; CHECK-NEXT:   .long   33554432                        # 0x2000000
; CHECK-NEXT:   .long   12
; CHECK-NEXT:   .long   0                               # BTF_KIND_PTR(id = 7)
; CHECK-NEXT:   .long   33554432                        # 0x2000000
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   49                              # BTF_KIND_STRUCT(id = 8)
; CHECK-NEXT:   .long   67108865                        # 0x4000001
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   53
; CHECK-NEXT:   .long   9
; CHECK-NEXT:   .long   0                               # 0x0
; CHECK-NEXT:   .long   55                              # BTF_KIND_INT(id = 9)
; CHECK-NEXT:   .long   16777216                        # 0x1000000
; CHECK-NEXT:   .long   4
; CHECK-NEXT:   .long   16777248                        # 0x1000020
; CHECK-NEXT:   .long   59                              # BTF_KIND_FUNC(id = 10)
; CHECK-NEXT:   .long   201326594                       # 0xc000002
; CHECK-NEXT:   .long   3
; CHECK-NEXT:   .long   64                              # BTF_KIND_TYPE_TAG(id = 11)
; CHECK-NEXT:   .long   301989888                       # 0x12000000
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   69                              # BTF_KIND_TYPE_TAG(id = 12)
; CHECK-NEXT:   .long   301989888                       # 0x12000000
; CHECK-NEXT:   .long   11

; CHECK:        .ascii  "test"                          # string offset=1
; CHECK:        .ascii  "map_value"                     # string offset=35
; CHECK:        .ascii  "ptr"                           # string offset=45
; CHECK:        .ascii  "foo"                           # string offset=49
; CHECK:        .byte   105                             # string offset=53
; CHECK:        .ascii  "int"                           # string offset=55
; CHECK:        .ascii  "func"                          # string offset=59
; CHECK:        .ascii  "tag2"                          # string offset=64
; CHECK:        .ascii  "tag1"                          # string offset=69

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !31 dso_local void @func(ptr noundef, ptr noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 5aa6dd1e09616a455377f3066d2034d3e8a073ba)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 5aa6dd1e09616a455377f3066d2034d3e8a073ba)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 11, type: !8, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 13, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: !1, line: 7, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !12, file: !1, line: 8, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, annotations: !24)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 32, elements: !17, annotations: !21)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !19, file: !1, line: 5, baseType: !20, size: 32)
!19 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, size: 32, elements: !17)
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !{!22, !23}
!22 = !{!"btf:type_tag", !"tag2"}
!23 = !{!"btf:type_tag", !"tag1"}
!24 = !{!25, !26}
!25 = !{!"btf_type_tag", !"tag2"}
!26 = !{!"btf_type_tag", !"tag1"}
!27 = !DILocation(line: 13, column: 9, scope: !7)
!28 = !DILocation(line: 13, column: 26, scope: !7)
!29 = !DILocation(line: 14, column: 9, scope: !7)
!30 = !DILocation(line: 15, column: 1, scope: !7)
!31 = !DISubprogram(name: "func", scope: !1, file: !1, line: 10, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !36)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !34, !35}
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!36 = !{!37, !38}
!37 = !DILocalVariable(arg: 1, scope: !31, file: !1, line: 10, type: !34)
!38 = !DILocalVariable(arg: 2, scope: !31, file: !1, line: 10, type: !35)
