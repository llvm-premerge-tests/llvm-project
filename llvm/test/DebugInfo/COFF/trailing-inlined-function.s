# RUN: llvm-mc -filetype=obj --triple=x86_64-pc-windows-msvc %s | llvm-readobj - --codeview --codeview-subsection-bytes | FileCheck %s

# Rust source to regenerate:
# #[no_mangle]
# extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
#     let x1 = x.unwrap();
#     let y1 = y.unwrap();
#     x1 + y1
# }
# $ rustc trailing-inlined-function.rs --crate-type cdylib --emit=asm -Copt-level=3 -Cpanic=abort -Cdebuginfo=2

# Validate that unwrap() was inlined.
# CHECK:       InlineSiteSym {
# CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
# CHECK-NEXT:    PtrParent: 0x0
# CHECK-NEXT:    PtrEnd: 0x0
# CHECK-NEXT:    Inlinee: unwrap (0x1023)
# CHECK-NEXT:    BinaryAnnotations [
# CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset1_1:]], LineOffset: 1}
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length1_1:]]
# CHECK-NEXT:      ChangeLineOffset: 2
# CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset1_2:]]
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
# CHECK-NEXT:      (Annotation Padding)
# CHECK:      InlineSiteSym {
# CHECK-NEXT:    Kind: S_INLINESITE (0x114D)
# CHECK-NEXT:    PtrParent: 0x0
# CHECK-NEXT:    PtrEnd: 0x0
# CHECK-NEXT:    Inlinee: unwrap (0x1023)
# CHECK-NEXT:    BinaryAnnotations [
# CHECK-NEXT:      ChangeCodeOffsetAndLineOffset: {CodeOffset: [[#%#x,Offset2_1:]], LineOffset: 1}
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,Length2_1:]]
# CHECK-NEXT:      ChangeLineOffset: 2
# CHECK-NEXT:      ChangeCodeOffset: [[#%#x,Offset2_2:]]
# CHECK-NEXT:      ChangeCodeLength: [[#%#x,]]
# CHECK-NEXT:      (Annotation Padding)

# Validate that basic blocks from an inlined function that are sunk below the rest of the function
# (namely bb1 and bb4 in this test) get the correct debug info.
# CHECK:       SubSectionType: Lines (0xF2)
# CHECK-NEXT:   SubSectionSize: [[#%#x,]]
# CHECK-NEXT:   SubSectionContents (
# CHECK-NEXT:     0000: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] [[#%.8X,]]
#                       Code starts at line 2
# CHECK-NEXT:     0010: [[#%.8X,]] [[#%.8X,]] [[#%.8X,]] 02000000
#                       The success paths for unwrap() (lines 3 & 4) are next.
# CHECK-NEXT:     0020: [[#%.2X,Offset1_1]]000000 03000000 [[#%.2X,Offset2_1]]000000 04000000
#                       Then the addition (line 5) and the end of the function (end-brace on line 6).
# CHECK-NEXT:     0030: [[#%.8X,]] 05000000 [[#%.8X,]] 06000000
#                       The failure paths for unwrap() (lines 3 & 4) are placed after the `ret` instruction.
# CHECK-NEXT:     0040: [[#%.2X,Offset1_1 + Length1_1 + Offset1_2]]000000 03000000 [[#%.2X,Offset2_1 + Length2_1 + Offset2_2]]000000 04000000
# CHECK-NOT:    SubSectionType: Lines (0xF2)

	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"trailing_inlined_function.3a6e73a087a7434a-cgu.0"
	.def	add_numbers;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",one_only,add_numbers
	.globl	add_numbers
	.p2align	4, 0x90
add_numbers:
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\llvm\\trailing-inlined-function.rs" "A63E3A719BDF505386FDB73BF86EC58591BDAC588181F0E423E724AEEC3E4852" 3
	.cv_loc	0 1 2 0
.seh_proc add_numbers
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
.Ltmp0:
	.cv_file	2 "/rustc/bc28abf92efc32f8f9312851bf8af38fbd23be42\\library\\core\\src\\option.rs" "7B702FA8D5AAEDC0CCA1EE32F30D5922BC11516B54D592279493A30457F918D9" 3
	.cv_inline_site_id 1 within 0 inlined_at 1 3 0
	.cv_loc	1 2 933 0
	cmpl	$0, (%rcx)
	je	.LBB0_1
.Ltmp1:
	.cv_inline_site_id 2 within 0 inlined_at 1 4 0
	.cv_loc	2 2 933 0
	cmpl	$0, (%rdx)
	je	.LBB0_4
.Ltmp2:
	.cv_loc	0 1 5 0
	movl	4(%rcx), %eax
.Ltmp3:
	addl	4(%rdx), %eax
.Ltmp4:
	.cv_loc	0 1 6 0
	addq	$40, %rsp
	retq
.Ltmp5:
.LBB0_1:
	.cv_loc	1 2 935 0
	leaq	__unnamed_1(%rip), %rcx
.Ltmp6:
	leaq	__unnamed_2(%rip), %r8
.Ltmp7:
	movl	$43, %edx
.Ltmp8:
	callq	_ZN4core9panicking5panic17hd083df7b722701afE
	ud2
.Ltmp9:
.LBB0_4:
	.cv_loc	2 2 935 0
	leaq	__unnamed_1(%rip), %rcx
.Ltmp10:
	leaq	__unnamed_3(%rip), %r8
.Ltmp11:
	movl	$43, %edx
.Ltmp12:
	callq	_ZN4core9panicking5panic17hd083df7b722701afE
	ud2
.Ltmp13:
.Lfunc_end0:
	.seh_endproc

	.section	.rdata,"dr",one_only,__unnamed_1
__unnamed_1:
	.ascii	"called `Option::unwrap()` on a `None` value"

	.section	.rdata,"dr",one_only,__unnamed_4
__unnamed_4:
	.ascii	"trailing-inlined-function.rs"

	.section	.rdata,"dr",one_only,__unnamed_2
	.p2align	3, 0x0
__unnamed_2:
	.quad	__unnamed_4
	.asciz	"\034\000\000\000\000\000\000\000\003\000\000\000\020\000\000"

	.section	.rdata,"dr",one_only,__unnamed_3
	.p2align	3, 0x0
__unnamed_3:
	.quad	__unnamed_4
	.asciz	"\034\000\000\000\000\000\000\000\004\000\000\000\020\000\000"

	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4
	.long	241
	.long	.Ltmp15-.Ltmp14
.Ltmp14:
	.short	.Ltmp17-.Ltmp16
.Ltmp16:
	.short	4353
	.long	0
	.byte	0
	.p2align	2, 0x0
.Ltmp17:
	.short	.Ltmp19-.Ltmp18
.Ltmp18:
	.short	4412
	.long	21
	.short	208
	.short	1
	.short	73
	.short	0
	.short	0
	.short	17000
	.short	0
	.short	0
	.short	0
	.asciz	"clang LLVM (rustc version 1.73.0-beta.3 (bc28abf92 2023-08-27))"
	.p2align	2, 0x0
.Ltmp19:
.Ltmp15:
	.p2align	2, 0x0
	.long	246
	.long	.Ltmp21-.Ltmp20
.Ltmp20:
	.long	0


	.long	4131
	.cv_filechecksumoffset	2
	.long	932


	.long	4131
	.cv_filechecksumoffset	2
	.long	932
.Ltmp21:
	.p2align	2, 0x0
	.section	.debug$S,"dr",associative,add_numbers
	.p2align	2, 0x0
	.long	4
	.long	241
	.long	.Ltmp23-.Ltmp22
.Ltmp22:
	.short	.Ltmp25-.Ltmp24
.Ltmp24:
	.short	4423
	.long	0
	.long	0
	.long	0
	.long	.Lfunc_end0-add_numbers
	.long	0
	.long	0
	.long	4136
	.secrel32	add_numbers
	.secidx	add_numbers
	.byte	128
	.asciz	"trailing_inlined_function::add_numbers"
	.p2align	2, 0x0
.Ltmp25:
	.short	.Ltmp27-.Ltmp26
.Ltmp26:
	.short	4114
	.long	40
	.long	0
	.long	0
	.long	0
	.long	0
	.short	0
	.long	1138688
	.p2align	2, 0x0
.Ltmp27:
	.short	.Ltmp29-.Ltmp28
.Ltmp28:
	.short	4414
	.long	4133
	.short	1
	.asciz	"x"
	.p2align	2, 0x0
.Ltmp29:
	.cv_def_range	 .Lfunc_begin0 .Ltmp6 .Ltmp9 .Ltmp10, reg, 330
	.short	.Ltmp31-.Ltmp30
.Ltmp30:
	.short	4414
	.long	4133
	.short	1
	.asciz	"y"
	.p2align	2, 0x0
.Ltmp31:
	.cv_def_range	 .Lfunc_begin0 .Ltmp8 .Ltmp9 .Ltmp12, reg, 331
	.short	.Ltmp33-.Ltmp32
.Ltmp32:
	.short	4414
	.long	116
	.short	0
	.asciz	"x1"
	.p2align	2, 0x0
.Ltmp33:
	.cv_def_range	 .Ltmp3 .Ltmp4, reg, 17
	.short	.Ltmp35-.Ltmp34
.Ltmp34:
	.short	4355
	.long	0
	.long	0
	.long	.Ltmp4-.Ltmp3
	.secrel32	.Ltmp3
	.secidx	.Lfunc_begin0
	.byte	0
	.p2align	2, 0x0
.Ltmp35:
	.short	.Ltmp37-.Ltmp36
.Ltmp36:
	.short	4414
	.long	116
	.short	256
	.asciz	"y1"
	.p2align	2, 0x0
.Ltmp37:
	.short	2
	.short	6
	.short	.Ltmp39-.Ltmp38
.Ltmp38:
	.short	4429
	.long	0
	.long	0
	.long	4131
	.cv_inline_linetable	1 2 932 .Lfunc_begin0 .Lfunc_end0
	.p2align	2, 0x0
.Ltmp39:
	.short	.Ltmp41-.Ltmp40
.Ltmp40:
	.short	4414
	.long	4100
	.short	256
	.asciz	"self"
	.p2align	2, 0x0
.Ltmp41:
	.short	2
	.short	4430
	.short	.Ltmp43-.Ltmp42
.Ltmp42:
	.short	4429
	.long	0
	.long	0
	.long	4131
	.cv_inline_linetable	2 2 932 .Lfunc_begin0 .Lfunc_end0
	.p2align	2, 0x0
.Ltmp43:
	.short	.Ltmp45-.Ltmp44
.Ltmp44:
	.short	4414
	.long	4100
	.short	256
	.asciz	"self"
	.p2align	2, 0x0
.Ltmp45:
	.short	2
	.short	4430
	.short	2
	.short	4431
.Ltmp23:
	.p2align	2, 0x0
	.cv_linetable	0, add_numbers, .Lfunc_end0
	.section	.debug$S,"dr"
	.long	241
	.long	.Ltmp47-.Ltmp46
.Ltmp46:
	.short	.Ltmp49-.Ltmp48
.Ltmp48:
	.short	4359
	.long	4105
	.byte	0x00, 0x00
	.asciz	"enum2$<core::option::Option<i32> >::Variant0::NAME"
	.p2align	2, 0x0
.Ltmp49:
	.short	.Ltmp51-.Ltmp50
.Ltmp50:
	.short	4359
	.long	117
	.byte	0x00, 0x00
	.asciz	"enum2$<core::option::Option<i32> >::Variant0::DISCR_EXACT"
	.p2align	2, 0x0
.Ltmp51:
	.short	.Ltmp53-.Ltmp52
.Ltmp52:
	.short	4359
	.long	4105
	.byte	0x01, 0x00
	.asciz	"enum2$<core::option::Option<i32> >::Variant1::NAME"
	.p2align	2, 0x0
.Ltmp53:
	.short	.Ltmp55-.Ltmp54
.Ltmp54:
	.short	4359
	.long	117
	.byte	0x01, 0x00
	.asciz	"enum2$<core::option::Option<i32> >::Variant1::DISCR_EXACT"
	.p2align	2, 0x0
.Ltmp55:
.Ltmp47:
	.p2align	2, 0x0
	.long	241
	.long	.Ltmp57-.Ltmp56
.Ltmp56:
	.short	.Ltmp59-.Ltmp58
.Ltmp58:
	.short	4360
	.long	117
	.asciz	"u32"
	.p2align	2, 0x0
.Ltmp59:
	.short	.Ltmp61-.Ltmp60
.Ltmp60:
	.short	4360
	.long	4100
	.asciz	"enum2$<core::option::Option<i32> >"
	.p2align	2, 0x0
.Ltmp61:
	.short	.Ltmp63-.Ltmp62
.Ltmp62:
	.short	4360
	.long	4108
	.asciz	"enum2$<core::option::Option<i32> >::Variant0"
	.p2align	2, 0x0
.Ltmp63:
	.short	.Ltmp65-.Ltmp64
.Ltmp64:
	.short	4360
	.long	4112
	.asciz	"enum2$<core::option::Option<i32> >::Variant1"
	.p2align	2, 0x0
.Ltmp65:
	.short	.Ltmp67-.Ltmp66
.Ltmp66:
	.short	4360
	.long	4115
	.asciz	"enum2$<core::option::Option<i32> >::None"
	.p2align	2, 0x0
.Ltmp67:
	.short	.Ltmp69-.Ltmp68
.Ltmp68:
	.short	4360
	.long	116
	.asciz	"i32"
	.p2align	2, 0x0
.Ltmp69:
	.short	.Ltmp71-.Ltmp70
.Ltmp70:
	.short	4360
	.long	4118
	.asciz	"enum2$<core::option::Option<i32> >::Some"
	.p2align	2, 0x0
.Ltmp71:
	.short	.Ltmp73-.Ltmp72
.Ltmp72:
	.short	4360
	.long	4126
	.asciz	"core::panic::location::Location"
	.p2align	2, 0x0
.Ltmp73:
	.short	.Ltmp75-.Ltmp74
.Ltmp74:
	.short	4360
	.long	32
	.asciz	"u8"
	.p2align	2, 0x0
.Ltmp75:
	.short	.Ltmp77-.Ltmp76
.Ltmp76:
	.short	4360
	.long	35
	.asciz	"usize"
	.p2align	2, 0x0
.Ltmp77:
	.short	.Ltmp79-.Ltmp78
.Ltmp78:
	.short	4360
	.long	4129
	.asciz	"ref$<str$>"
	.p2align	2, 0x0
.Ltmp79:
.Ltmp57:
	.p2align	2, 0x0
	.cv_filechecksums
	.cv_stringtable
	.long	241
	.long	.Ltmp81-.Ltmp80
.Ltmp80:
	.short	.Ltmp83-.Ltmp82
.Ltmp82:
	.short	4428
	.long	4140
	.p2align	2, 0x0
.Ltmp83:
.Ltmp81:
	.p2align	2, 0x0
	.section	.debug$T,"dr"
	.p2align	2, 0x0
	.long	4
	.short	0x52
	.short	0x1506
	.short	0x0
	.short	0x280
	.long	0x0
	.short	0x0
	.asciz	"enum2$<core::option::Option<i32> >"
	.asciz	"997a8e6fb304af087b53852b998af779"
	.byte	242
	.byte	241
	.short	0x62
	.short	0x1505
	.short	0x0
	.short	0x288
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"enum2$<core::option::Option<i32> >::Variant0"
	.asciz	"a8260c068e3cafa8a1732c83fb160db5"
	.short	0x62
	.short	0x1505
	.short	0x0
	.short	0x288
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"enum2$<core::option::Option<i32> >::Variant1"
	.asciz	"fca09fe9e9a1b7ab846056b0c103408f"
	.short	0x3a
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x1001
	.short	0x0
	.asciz	"variant0"
	.byte	241
	.short	0x150d
	.short	0x3
	.long	0x1002
	.short	0x0
	.asciz	"variant1"
	.byte	241
	.short	0x150d
	.short	0x3
	.long	0x75
	.short	0x0
	.asciz	"tag"
	.byte	242
	.byte	241
	.short	0x52
	.short	0x1506
	.short	0x3
	.short	0x600
	.long	0x1003
	.short	0x8
	.asciz	"enum2$<core::option::Option<i32> >"
	.asciz	"997a8e6fb304af087b53852b998af779"
	.byte	242
	.byte	241
	.short	0x12
	.short	0x1605
	.long	0x0
	.asciz	"\\<unknown>"
	.byte	241
	.short	0xe
	.short	0x1606
	.long	0x1004
	.long	0x1005
	.long	0x0
	.short	0x5e
	.short	0x1505
	.short	0x0
	.short	0x288
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"enum2$<core::option::Option<i32> >::None"
	.asciz	"6509d49bc5c48d7538f52454e961241e"
	.short	0x1a
	.short	0x1203
	.short	0x1502
	.short	0x3
	.short	0x0
	.asciz	"None"
	.byte	241
	.short	0x1502
	.short	0x3
	.short	0x1
	.asciz	"Some"
	.byte	241
	.short	0x42
	.short	0x1507
	.short	0x2
	.short	0x8
	.long	0x75
	.long	0x1008
	.asciz	"enum2$<core::option::Option<i32> >::VariantNames"
	.byte	243
	.byte	242
	.byte	241
	.short	0xe
	.short	0x1606
	.long	0x1009
	.long	0x1005
	.long	0x0
	.short	0x36
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x1007
	.short	0x0
	.asciz	"value"
	.short	0x150e
	.short	0x3
	.long	0x1009
	.asciz	"NAME"
	.byte	243
	.byte	242
	.byte	241
	.short	0x150e
	.short	0x3
	.long	0x75
	.asciz	"DISCR_EXACT"
	.short	0x62
	.short	0x1505
	.short	0x3
	.short	0x208
	.long	0x100b
	.long	0x0
	.long	0x0
	.short	0x8
	.asciz	"enum2$<core::option::Option<i32> >::Variant0"
	.asciz	"a8260c068e3cafa8a1732c83fb160db5"
	.short	0xe
	.short	0x1606
	.long	0x100c
	.long	0x1005
	.long	0x0
	.short	0x5e
	.short	0x1505
	.short	0x0
	.short	0x288
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"enum2$<core::option::Option<i32> >::Some"
	.asciz	"35e7327d29891293cef33620fd88f4bc"
	.short	0x36
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x100e
	.short	0x0
	.asciz	"value"
	.short	0x150e
	.short	0x3
	.long	0x1009
	.asciz	"NAME"
	.byte	243
	.byte	242
	.byte	241
	.short	0x150e
	.short	0x3
	.long	0x75
	.asciz	"DISCR_EXACT"
	.short	0x62
	.short	0x1505
	.short	0x3
	.short	0x208
	.long	0x100f
	.long	0x0
	.long	0x0
	.short	0x8
	.asciz	"enum2$<core::option::Option<i32> >::Variant1"
	.asciz	"fca09fe9e9a1b7ab846056b0c103408f"
	.short	0xe
	.short	0x1606
	.long	0x1010
	.long	0x1005
	.long	0x0
	.short	0x2
	.short	0x1203
	.short	0x5e
	.short	0x1505
	.short	0x0
	.short	0x208
	.long	0x1012
	.long	0x0
	.long	0x0
	.short	0x8
	.asciz	"enum2$<core::option::Option<i32> >::None"
	.asciz	"6509d49bc5c48d7538f52454e961241e"
	.short	0xe
	.short	0x1606
	.long	0x1013
	.long	0x1005
	.long	0x0
	.short	0x12
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x74
	.short	0x4
	.asciz	"__0"
	.byte	242
	.byte	241
	.short	0x5e
	.short	0x1505
	.short	0x1
	.short	0x208
	.long	0x1015
	.long	0x0
	.long	0x0
	.short	0x8
	.asciz	"enum2$<core::option::Option<i32> >::Some"
	.asciz	"35e7327d29891293cef33620fd88f4bc"
	.short	0xe
	.short	0x1606
	.long	0x1016
	.long	0x1005
	.long	0x0
	.short	0x56
	.short	0x1505
	.short	0x0
	.short	0x280
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"core::panic::location::Location"
	.asciz	"257317645c30b8a523f5e6171a087f5a"
	.byte	241
	.short	0xa
	.short	0x1002
	.long	0x1018
	.long	0x1000c
	.short	0xe
	.short	0x1201
	.long	0x2
	.long	0x1000
	.long	0x1019
	.short	0x1a
	.short	0x1009
	.long	0x74
	.long	0x1000
	.long	0x0
	.byte	0x0
	.byte	0x0
	.short	0x2
	.long	0x101a
	.long	0x0
	.short	0x42
	.short	0x1505
	.short	0x0
	.short	0x280
	.long	0x0
	.long	0x0
	.long	0x0
	.short	0x0
	.asciz	"ref$<str$>"
	.asciz	"857c99401054bcaa39f98e6e0c6d74b"
	.byte	243
	.byte	242
	.byte	241
	.short	0x32
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x101c
	.short	0x0
	.asciz	"file"
	.byte	241
	.short	0x150d
	.short	0x3
	.long	0x75
	.short	0x10
	.asciz	"line"
	.byte	241
	.short	0x150d
	.short	0x3
	.long	0x75
	.short	0x14
	.asciz	"col"
	.byte	242
	.byte	241
	.short	0x56
	.short	0x1505
	.short	0x3
	.short	0x200
	.long	0x101d
	.long	0x0
	.long	0x0
	.short	0x18
	.asciz	"core::panic::location::Location"
	.asciz	"257317645c30b8a523f5e6171a087f5a"
	.byte	241
	.short	0xe
	.short	0x1606
	.long	0x101e
	.long	0x1005
	.long	0x0
	.short	0x2a
	.short	0x1203
	.short	0x150d
	.short	0x3
	.long	0x620
	.short	0x0
	.asciz	"data_ptr"
	.byte	241
	.short	0x150d
	.short	0x3
	.long	0x23
	.short	0x8
	.asciz	"length"
	.byte	243
	.byte	242
	.byte	241
	.short	0x42
	.short	0x1505
	.short	0x2
	.short	0x200
	.long	0x1020
	.long	0x0
	.long	0x0
	.short	0x10
	.asciz	"ref$<str$>"
	.asciz	"857c99401054bcaa39f98e6e0c6d74b"
	.byte	243
	.byte	242
	.byte	241
	.short	0xe
	.short	0x1606
	.long	0x1021
	.long	0x1005
	.long	0x0
	.short	0x12
	.short	0x1602
	.long	0x1000
	.long	0x101b
	.asciz	"unwrap"
	.byte	241
	.short	0x22
	.short	0x1605
	.long	0x0
	.asciz	"trailing_inlined_function"
	.byte	242
	.byte	241
	.short	0xa
	.short	0x1002
	.long	0x1000
	.long	0x1000c
	.short	0xe
	.short	0x1201
	.long	0x2
	.long	0x1025
	.long	0x1025
	.short	0xe
	.short	0x1008
	.long	0x74
	.byte	0x0
	.byte	0x0
	.short	0x2
	.long	0x1026
	.short	0x16
	.short	0x1601
	.long	0x1024
	.long	0x1027
	.asciz	"add_numbers"
	.short	0xe
	.short	0x1605
	.long	0x0
	.asciz	"C:\\llvm"
	.short	0x56
	.short	0x1605
	.long	0x0
	.asciz	"trailing-inlined-function.rs\\@\\trailing_inlined_function.3a6e73a087a7434a-cgu.0"
	.short	0xa
	.short	0x1605
	.long	0x0
	.byte	0
	.byte	243
	.byte	242
	.byte	241
	.short	0x1a
	.short	0x1603
	.short	0x5
	.long	0x1029
	.long	0x0
	.long	0x102a
	.long	0x102b
	.long	0x0
	.byte	242
	.byte	241
