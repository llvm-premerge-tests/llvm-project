# REQUIRES: asserts
# RUN: llvm-mc -triple=riscv64-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -phony-externals -debug-only=jitlink %t 2>&1 | \
# RUN:   FileCheck %s

# Check that splitting of eh-frame sections works.
#
# CHECK: DWARFRecordSectionSplitter: Processing .eh_frame...
# CHECK:  Processing block at
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK:    Processing CFI record at
# CHECK:      Extracted {{.*}} section = .eh_frame
# CHECK: EHFrameEdgeFixer: Processing .eh_frame in "{{.*}}"...
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is CIE
# CHECK:   Processing block at
# CHECK:     Processing CFI record at
# CHECK:       Record is FDE
# CHECK:         Adding edge at {{.*}} to CIE at: {{.*}}
# CHECK:         Existing edge at {{.*}} to PC begin at {{.*}}
# CHECK:         Adding keep-alive edge from target at {{.*}} to FDE at {{.*}}

	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0"
	.file	"catch.cpp"
	.globl	main                            # -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0
# %bb.0:
	addi	sp, sp, -16
	.cfi_def_cfa_offset 16
	sd	ra, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	li	a0, 4
	call	__cxa_allocate_exception@plt
	li	a1, 1
	sw	a1, 0(a0)
.Ltmp0:
.Lpcrel_hi0:
	auipc	a1, %got_pcrel_hi(_ZTIi)
	ld	a1, %pcrel_lo(.Lpcrel_hi0)(a1)
	li	a2, 0
	call	__cxa_throw@plt
.Ltmp1:
# %bb.1:
.LBB0_2:
.Ltmp2:
	call	__cxa_begin_catch@plt
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(.Lstr)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi1)
	call	puts@plt
	call	__cxa_end_catch@plt
	li	a0, 0
	ld	ra, 8(sp)                       # 8-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	155                             # @TType Encoding = indirect pcrel sdata4
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	3                               # Call site Encoding = udata4
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.word	.Lfunc_begin0-.Lfunc_begin0     # >> Call Site 1 <<
	.word	.Ltmp0-.Lfunc_begin0            #   Call between .Lfunc_begin0 and .Ltmp0
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.word	.Ltmp0-.Lfunc_begin0            # >> Call Site 2 <<
	.word	.Ltmp1-.Ltmp0                   #   Call between .Ltmp0 and .Ltmp1
	.word	.Ltmp2-.Lfunc_begin0            #     jumps to .Ltmp2
	.byte	1                               #   On action: 1
	.word	.Ltmp1-.Lfunc_begin0            # >> Call Site 3 <<
	.word	.Lfunc_end0-.Ltmp1              #   Call between .Ltmp1 and .Lfunc_end0
	.word	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2, 0x0
                                        # >> Catch TypeInfos <<
	.word	0                               # TypeInfo 1
.Lttbase0:
	.p2align	2, 0x0
                                        # -- End function
	.type	.Lstr,@object                   # @str
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lstr:
	.asciz	"Caught!"
	.size	.Lstr, 8

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"clang version 18.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZTIi
