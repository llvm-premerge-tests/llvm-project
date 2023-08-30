; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview --codeview-subsection-bytes | FileCheck %s

; Rust source to regenerate:
; #[no_mangle]
; extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
;     let x1 = x.unwrap();
;     let y1 = y.unwrap();
;     x1 + y1
; }
; $ rustc trailing-inlined-function.rs --crate-type cdylib --emit=llvm-ir -Copt-level=3 -Cpanic=abort -Cdebuginfo=2

; Validate that unwrap() was inlined.
; CHECK:       InlineSiteSym {
; CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
; CHECK-NEXT:    PtrParent: 0x0
; CHECK-NEXT:    PtrEnd: 0x0
; CHECK-NEXT:    Inlinee: unwrap (0x1023)
; CHECK-NEXT:    BinaryAnnotations [
; CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset1_1:]], LineOffset: 1}
; CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length1_1:]]
; CHECK-NEXT:      ChangeLineOffset: 2
; CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset1_2:]]
; CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
; CHECK-NEXT:      (Annotation Padding)
; CHECK:      InlineSiteSym {
; CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
; CHECK-NEXT:    PtrParent: 0x0
; CHECK-NEXT:    PtrEnd: 0x0
; CHECK-NEXT:    Inlinee: unwrap (0x1023)
; CHECK-NEXT:    BinaryAnnotations [
; CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset2_1:]], LineOffset: 1}
; CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length2_1:]]
; CHECK-NEXT:      ChangeLineOffset: 2
; CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset2_2:]]
; CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
; CHECK-NEXT:      (Annotation Padding)

; Validate that basic blocks from an inlined function that are sunk below the rest of the function
; (namely bb1 and bb4 in this test) get the correct debug info.
; CHECK:       SubSectionType: Lines (0xF2)
; CHECK-NEXT:   SubSectionSize: [[#%#x,]]
; CHECK-NEXT:   SubSectionContents (
; CHECK-NEXT:     0000: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] [[#%.8X,]]
;                       Code starts at line 2
; CHECK-NEXT:     0010: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] 02000000
;                       The success paths for unwrap() (lines 3 & 4) are next.
; CHECK-NEXT:     0020: [[#%.2X,Offset1_1]]000000 03000000 [[#%.2X,Offset2_1]]000000 04000000
;                       Then the addition (line 5) and the end of the function (end-brace on line 6).
; CHECK-NEXT:     0030: [[#%.8X,]] 05000000 [[#%.8X,]] 06000000
;                       The failure paths for unwrap() (lines 3 & 4) are placed after the `ret` instruction.
; CHECK-NEXT:     0040: [[#%.2X,Offset1_1 + Length1_1 + Offset1_2]]000000 03000000 [[#%.2X,Offset2_1 + Length2_1 + Offset2_2]]000000 04000000
; CHECK-NOT:    SubSectionType: Lines (0xF2)

; ModuleID = 'trailing_inlined_function.3a6e73a087a7434a-cgu.0'
source_filename = "trailing_inlined_function.3a6e73a087a7434a-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@alloc_5f55955de67e57c79064b537689facea = private unnamed_addr constant <{ [43 x i8] }> <{ [43 x i8] c"called `Option::unwrap()` on a `None` value" }>, align 1
@alloc_7532843ef034b9947726fe35054b1579 = private unnamed_addr constant <{ [28 x i8] }> <{ [28 x i8] c"trailing-inlined-function.rs" }>, align 1
@alloc_896f7e06def4b3514dbfbeea7925faf6 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_7532843ef034b9947726fe35054b1579, [16 x i8] c"\1C\00\00\00\00\00\00\00\03\00\00\00\10\00\00\00" }>, align 8
@alloc_7f0fe7c4d58560ef64784c94c01b8739 = private unnamed_addr constant <{ ptr, [16 x i8] }> <{ ptr @alloc_7532843ef034b9947726fe35054b1579, [16 x i8] c"\1C\00\00\00\00\00\00\00\04\00\00\00\10\00\00\00" }>, align 8

; Function Attrs: nounwind uwtable
define noundef i32 @add_numbers(ptr noalias nocapture noundef readonly align 4 dereferenceable(8) %x, ptr noalias nocapture noundef readonly align 4 dereferenceable(8) %y) unnamed_addr #0 !dbg !38 {
start:
  call void @llvm.dbg.value(metadata ptr %x, metadata !45, metadata !DIExpression()), !dbg !51
  call void @llvm.dbg.value(metadata ptr %y, metadata !46, metadata !DIExpression()), !dbg !51
  %0 = load i32, ptr %x, align 4, !dbg !52, !range !53, !noundef !16
  %1 = getelementptr inbounds { i32, i32 }, ptr %x, i64 0, i32 1, !dbg !52
  %2 = load i32, ptr %1, align 4, !dbg !52
  call void @llvm.dbg.value(metadata i32 %0, metadata !54, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !81
  call void @llvm.dbg.value(metadata i32 %2, metadata !54, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !81
  %trunc.not = icmp eq i32 %0, 0, !dbg !82
  br i1 %trunc.not, label %bb1, label %bb3, !dbg !82

bb1:                                              ; preds = %start
; call core::panicking::panic
  tail call void @_ZN4core9panicking5panic17hd083df7b722701afE(ptr noalias noundef nonnull readonly align 1 @alloc_5f55955de67e57c79064b537689facea, i64 noundef 43, ptr noalias noundef nonnull readonly align 8 dereferenceable(24) @alloc_896f7e06def4b3514dbfbeea7925faf6) #3, !dbg !83
  unreachable, !dbg !83

bb3:                                              ; preds = %start
  call void @llvm.dbg.value(metadata i32 %2, metadata !47, metadata !DIExpression()), !dbg !84
  %3 = load i32, ptr %y, align 4, !dbg !85, !range !53, !noundef !16
  call void @llvm.dbg.value(metadata i32 %3, metadata !86, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !91
  call void @llvm.dbg.value(metadata i32 poison, metadata !86, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !91
  %trunc2.not = icmp eq i32 %3, 0, !dbg !92
  br i1 %trunc2.not, label %bb4, label %bb5, !dbg !92

bb4:                                              ; preds = %bb3
; call core::panicking::panic
  tail call void @_ZN4core9panicking5panic17hd083df7b722701afE(ptr noalias noundef nonnull readonly align 1 @alloc_5f55955de67e57c79064b537689facea, i64 noundef 43, ptr noalias noundef nonnull readonly align 8 dereferenceable(24) @alloc_7f0fe7c4d58560ef64784c94c01b8739) #3, !dbg !93
  unreachable, !dbg !93

bb5:                                              ; preds = %bb3
  %4 = getelementptr inbounds { i32, i32 }, ptr %y, i64 0, i32 1, !dbg !85
  %5 = load i32, ptr %4, align 4, !dbg !85
  call void @llvm.dbg.value(metadata i32 %5, metadata !86, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !91
  call void @llvm.dbg.value(metadata i32 poison, metadata !49, metadata !DIExpression()), !dbg !94
  %_0 = add i32 %5, %2, !dbg !95
  ret i32 %_0, !dbg !96
}

; core::panicking::panic
; Function Attrs: cold noinline noreturn nounwind uwtable
declare void @_ZN4core9panicking5panic17hd083df7b722701afE(ptr noalias noundef nonnull readonly align 1, i64 noundef, ptr noalias noundef readonly align 8 dereferenceable(24)) unnamed_addr #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" }
attributes #1 = { cold noinline noreturn nounwind uwtable "target-cpu"="x86-64" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{!"rustc version 1.73.0-beta.3 (bc28abf92 2023-08-27)"}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.73.0-beta.3 (bc28abf92 2023-08-27))", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, splitDebugInlining: false)
!5 = !DIFile(filename: "trailing-inlined-function.rs\\@\\trailing_inlined_function.3a6e73a087a7434a-cgu.0", directory: "C:\\llvm")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "VariantNames", scope: !9, file: !8, baseType: !23, size: 32, align: 32, flags: DIFlagEnumClass, elements: !35)
!8 = !DIFile(filename: "<unknown>", directory: "")
!9 = !DICompositeType(tag: DW_TAG_union_type, name: "enum2$<core::option::Option<i32> >", file: !8, size: 64, align: 32, elements: !10, templateParams: !16, identifier: "997a8e6fb304af087b53852b998af779")
!10 = !{!11, !25, !34}
!11 = !DIDerivedType(tag: DW_TAG_member, name: "variant0", scope: !9, file: !8, baseType: !12, size: 64, align: 32)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "Variant0", scope: !9, file: !8, size: 64, align: 32, elements: !13, templateParams: !16, identifier: "a8260c068e3cafa8a1732c83fb160db5")
!13 = !{!14, !21, !22}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !12, file: !8, baseType: !15, size: 64, align: 32)
!15 = !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !9, file: !8, size: 64, align: 32, elements: !16, templateParams: !17, identifier: "6509d49bc5c48d7538f52454e961241e")
!16 = !{}
!17 = !{!18}
!18 = !DITemplateTypeParameter(name: "T", type: !19)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "i32", file: !8, baseType: !20)
!20 = !DIBasicType(name: "__int32", size: 32, encoding: DW_ATE_signed)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "NAME", scope: !12, file: !8, baseType: !7, align: 32, flags: DIFlagStaticMember, extraData: i64 0)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "DISCR_EXACT", scope: !12, file: !8, baseType: !23, align: 32, flags: DIFlagStaticMember, extraData: i64 0)
!23 = !DIDerivedType(tag: DW_TAG_typedef, name: "u32", file: !8, baseType: !24)
!24 = !DIBasicType(name: "unsigned __int32", size: 32, encoding: DW_ATE_unsigned)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "variant1", scope: !9, file: !8, baseType: !26, size: 64, align: 32)
!26 = !DICompositeType(tag: DW_TAG_structure_type, name: "Variant1", scope: !9, file: !8, size: 64, align: 32, elements: !27, templateParams: !16, identifier: "fca09fe9e9a1b7ab846056b0c103408f")
!27 = !{!28, !32, !33}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !26, file: !8, baseType: !29, size: 64, align: 32)
!29 = !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !9, file: !8, size: 64, align: 32, elements: !30, templateParams: !17, identifier: "35e7327d29891293cef33620fd88f4bc")
!30 = !{!31}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !29, file: !8, baseType: !19, size: 32, align: 32, offset: 32)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "NAME", scope: !26, file: !8, baseType: !7, align: 32, flags: DIFlagStaticMember, extraData: i64 1)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "DISCR_EXACT", scope: !26, file: !8, baseType: !23, align: 32, flags: DIFlagStaticMember, extraData: i64 1)
!34 = !DIDerivedType(tag: DW_TAG_member, name: "tag", scope: !9, file: !8, baseType: !23, size: 32, align: 32)
!35 = !{!36, !37}
!36 = !DIEnumerator(name: "None", value: 0, isUnsigned: true)
!37 = !DIEnumerator(name: "Some", value: 1, isUnsigned: true)
!38 = distinct !DISubprogram(name: "add_numbers", scope: !40, file: !39, line: 2, type: !41, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !4, templateParams: !16, retainedNodes: !44)
!39 = !DIFile(filename: "trailing-inlined-function.rs", directory: "C:\\llvm", checksumkind: CSK_SHA256, checksum: "52214be4e1ee44649b02f0e0de238615473fd7e258bd506638380d2be2d5b17d")
!40 = !DINamespace(name: "trailing_inlined_function", scope: null)
!41 = !DISubroutineType(types: !42)
!42 = !{!19, !43, !43}
!43 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ref$<enum2$<core::option::Option<i32> > >", baseType: !9, size: 64, align: 64, dwarfAddressSpace: 0)
!44 = !{!45, !46, !47, !49}
!45 = !DILocalVariable(name: "x", arg: 1, scope: !38, file: !39, line: 2, type: !43)
!46 = !DILocalVariable(name: "y", arg: 2, scope: !38, file: !39, line: 2, type: !43)
!47 = !DILocalVariable(name: "x1", scope: !48, file: !39, line: 3, type: !19, align: 4)
!48 = distinct !DILexicalBlock(scope: !38, file: !39, line: 3)
!49 = !DILocalVariable(name: "y1", scope: !50, file: !39, line: 4, type: !19, align: 4)
!50 = distinct !DILexicalBlock(scope: !48, file: !39, line: 4)
!51 = !DILocation(line: 0, scope: !38)
!52 = !DILocation(line: 3, scope: !38)
!53 = !{i32 0, i32 2}
!54 = !DILocalVariable(name: "self", scope: !55, file: !56, line: 932, type: !9, align: 4)
!55 = distinct !DISubprogram(name: "unwrap<i32>", linkageName: "_ZN4core6option15Option$LT$T$GT$6unwrap17he334e4e23537a1b6E", scope: !9, file: !56, line: 932, type: !57, scopeLine: 932, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !4, templateParams: !17, declaration: !77, retainedNodes: !78)
!56 = !DIFile(filename: "/rustc/bc28abf92efc32f8f9312851bf8af38fbd23be42\\library\\core\\src\\option.rs", directory: "", checksumkind: CSK_SHA256, checksum: "7b702fa8d5aaedc0cca1ee32f30d5922bc11516b54d592279493a30457f918d9")
!57 = !DISubroutineType(types: !58)
!58 = !{!19, !9, !59}
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "ref$<core::panic::location::Location>", baseType: !60, size: 64, align: 64, dwarfAddressSpace: 0)
!60 = !DICompositeType(tag: DW_TAG_structure_type, name: "Location", scope: !61, file: !8, size: 192, align: 64, elements: !64, templateParams: !16, identifier: "257317645c30b8a523f5e6171a087f5a")
!61 = !DINamespace(name: "location", scope: !62)
!62 = !DINamespace(name: "panic", scope: !63)
!63 = !DINamespace(name: "core", scope: null)
!64 = !{!65, !75, !76}
!65 = !DIDerivedType(tag: DW_TAG_member, name: "file", scope: !60, file: !8, baseType: !66, size: 128, align: 64)
!66 = !DICompositeType(tag: DW_TAG_structure_type, name: "ref$<str$>", file: !8, size: 128, align: 64, elements: !67, templateParams: !16, identifier: "857c99401054bcaa39f98e6e0c6d74b")
!67 = !{!68, !72}
!68 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !66, file: !8, baseType: !69, size: 64, align: 64)
!69 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !70, size: 64, align: 64, dwarfAddressSpace: 0)
!70 = !DIDerivedType(tag: DW_TAG_typedef, name: "u8", file: !8, baseType: !71)
!71 = !DIBasicType(name: "unsigned __int8", size: 8, encoding: DW_ATE_unsigned)
!72 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !66, file: !8, baseType: !73, size: 64, align: 64, offset: 64)
!73 = !DIDerivedType(tag: DW_TAG_typedef, name: "usize", file: !8, baseType: !74)
!74 = !DIBasicType(name: "size_t", size: 64, encoding: DW_ATE_unsigned)
!75 = !DIDerivedType(tag: DW_TAG_member, name: "line", scope: !60, file: !8, baseType: !23, size: 32, align: 32, offset: 128)
!76 = !DIDerivedType(tag: DW_TAG_member, name: "col", scope: !60, file: !8, baseType: !23, size: 32, align: 32, offset: 160)
!77 = !DISubprogram(name: "unwrap<i32>", linkageName: "_ZN4core6option15Option$LT$T$GT$6unwrap17he334e4e23537a1b6E", scope: !9, file: !56, line: 932, type: !57, scopeLine: 932, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagOptimized, templateParams: !17)
!78 = !{!54, !79}
!79 = !DILocalVariable(name: "val", scope: !80, file: !56, line: 934, type: !19, align: 4)
!80 = distinct !DILexicalBlock(scope: !55, file: !56, line: 934)
!81 = !DILocation(line: 0, scope: !55, inlinedAt: !52)
!82 = !DILocation(line: 933, scope: !55, inlinedAt: !52)
!83 = !DILocation(line: 935, scope: !55, inlinedAt: !52)
!84 = !DILocation(line: 0, scope: !48)
!85 = !DILocation(line: 4, scope: !48)
!86 = !DILocalVariable(name: "self", scope: !87, file: !56, line: 932, type: !9, align: 4)
!87 = distinct !DISubprogram(name: "unwrap<i32>", linkageName: "_ZN4core6option15Option$LT$T$GT$6unwrap17he334e4e23537a1b6E", scope: !9, file: !56, line: 932, type: !57, scopeLine: 932, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !4, templateParams: !17, declaration: !77, retainedNodes: !88)
!88 = !{!86, !89}
!89 = !DILocalVariable(name: "val", scope: !90, file: !56, line: 934, type: !19, align: 4)
!90 = distinct !DILexicalBlock(scope: !87, file: !56, line: 934)
!91 = !DILocation(line: 0, scope: !87, inlinedAt: !85)
!92 = !DILocation(line: 933, scope: !87, inlinedAt: !85)
!93 = !DILocation(line: 935, scope: !87, inlinedAt: !85)
!94 = !DILocation(line: 0, scope: !50)
!95 = !DILocation(line: 5, scope: !50)
!96 = !DILocation(line: 6, scope: !38)
