# RUN: llvm-mc -filetype=obj -triple=ppc %s -o %t
# RUN: llvm-readelf -r %t | FileCheck %s 

# CHECK: Relocation section '.rela.debug_info' at offset 0x118 contains 1 entries:
# CHECK-NEXT: Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# CHECK-NEXT:00000013  0000024e R_PPC_DTPREL32         00000000   tls_rtp_var + 8000

	.text
	.globl	foo                             # -- Begin function foo
	.p2align	2
	.type	foo,@function
foo:                                    # @foo
# %bb.0:                                # %entry
	stwu 1, -16(1)
	stw 31, 12(1)
	mr	31, 1
	addis 3, 2, tls_rtp_var@tprel@ha
	addi 4, 3, tls_rtp_var@tprel@l
	lwz 3, 0(4)
	addi 3, 3, 1
	stw 3, 0(4)
	lis 4, my_global_var@ha
	lwz 3, my_global_var@l(4)
	addi 3, 3, 1
	stw 3, my_global_var@l(4)
	lwz 31, 12(1)
	addi 1, 1, 16
	blr

	.type	tls_rtp_var,@object             # @tls_rtp_var
	.section	.tdata,"awT",@progbits
	.globl	tls_rtp_var
	.p2align	2, 0x0
tls_rtp_var:
	.long	5                               # 0x5
	.size	tls_rtp_var, 4

	.type	my_global_var,@object           # @my_global_var
	.data
	.globl	my_global_var
	.p2align	2, 0x0
my_global_var:
	.long	7                               # 0x7
	.size	my_global_var, 4

	.section	.debug_info,"",@progbits
.Lcu_begin0:
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.byte	0                               # DW_AT_producer
	.short	29                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.byte	2                               # DW_AT_comp_dir
	.byte	1                               # DW_AT_low_pc
	.byte	3                               # DW_AT_name
	.long	50                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	6                               # DW_AT_location
	.byte	12
	.long	tls_rtp_var@DTPREL+32768
	.byte	224
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	5                               # DW_AT_name
	.long	50                              # DW_AT_type
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
