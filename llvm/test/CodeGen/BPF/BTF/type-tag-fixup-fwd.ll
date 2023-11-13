; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;   struct foo;
;   struct map_value {
;           struct foo __tag2 __tag1 *ptr;
;   };
;   void func(struct map_value *);
;   void test(void)
;   {
;           struct map_value v = {};
;           func(&v);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm test.c

%struct.map_value = type { ptr }

; Function Attrs: nounwind
define dso_local void @test() local_unnamed_addr #0 !dbg !7 {
entry:
  %v = alloca %struct.map_value, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %v) #4, !dbg !23
  call void @llvm.dbg.declare(metadata ptr %v, metadata !11, metadata !DIExpression()), !dbg !24
  store i64 0, ptr %v, align 8, !dbg !24
  call void @func(ptr noundef nonnull %v) #4, !dbg !25
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %v) #4, !dbg !26
  ret void, !dbg !26
}

; CHECK:        .long   0                               # BTF_KIND_FUNC_PROTO(id = 1)
; CHECK-NEXT:   .long   218103808                       # 0xd000000
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   1                               # BTF_KIND_FUNC(id = 2)
; CHECK-NEXT:   .long   201326593                       # 0xc000001
; CHECK-NEXT:   .long   1
; CHECK-NEXT:   .long   0                               # BTF_KIND_FUNC_PROTO(id = 3)
; CHECK-NEXT:   .long   218103809                       # 0xd000001
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   4
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
; CHECK-NEXT:   .long   9
; CHECK-NEXT:   .long   49                              # BTF_KIND_FWD(id = 7)
; CHECK-NEXT:   .long   117440512                       # 0x7000000
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .long   53                              # BTF_KIND_TYPE_TAG(id = 8)
; CHECK-NEXT:   .long   301989888                       # 0x12000000
; CHECK-NEXT:   .long   7
; CHECK-NEXT:   .long   58                              # BTF_KIND_TYPE_TAG(id = 9)
; CHECK-NEXT:   .long   301989888                       # 0x12000000
; CHECK-NEXT:   .long   8
; CHECK-NEXT:   .long   63                              # BTF_KIND_FUNC(id = 10)
; CHECK-NEXT:   .long   201326594                       # 0xc000002
; CHECK-NEXT:   .long   3

; CHECK:        .ascii  "test"                          # string offset=1
; CHECK:        .ascii  "map_value"                     # string offset=35
; CHECK:        .ascii  "ptr"                           # string offset=45
; CHECK:        .ascii  "foo"                           # string offset=49
; CHECK:        .ascii  "tag2"                          # string offset=53
; CHECK:        .ascii  "tag1"                          # string offset=58
; CHECK:        .ascii  "func"                          # string offset=63

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare !dbg !27 dso_local void @func(ptr noundef) local_unnamed_addr #3

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
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 9, type: !8, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 11, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: !1, line: 5, size: 64, elements: !13)
!13 = !{!14}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !12, file: !1, line: 6, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, annotations: !20)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !1, line: 4, flags: DIFlagFwdDecl, annotations: !17)
!17 = !{!18, !19}
!18 = !{!"btf:type_tag", !"tag2"}
!19 = !{!"btf:type_tag", !"tag1"}
!20 = !{!21, !22}
!21 = !{!"btf_type_tag", !"tag2"}
!22 = !{!"btf_type_tag", !"tag1"}
!23 = !DILocation(line: 11, column: 9, scope: !7)
!24 = !DILocation(line: 11, column: 26, scope: !7)
!25 = !DILocation(line: 12, column: 9, scope: !7)
!26 = !DILocation(line: 13, column: 1, scope: !7)
!27 = !DISubprogram(name: "func", scope: !1, file: !1, line: 8, type: !28, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !31)
!28 = !DISubroutineType(types: !29)
!29 = !{null, !30}
!30 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!31 = !{!32}
!32 = !DILocalVariable(arg: 1, scope: !27, file: !1, line: 8, type: !30)
