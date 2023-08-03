# RUN: llvm-mc -filetype=obj -triple=ppc %s -o %t
# RUN: llvm-readelf -r %t | FileCheck %s 

# CHECK: Relocation section '.rela.debug_info' at offset 0x28c contains 5 entries:
# CHECK-NEXT:  Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# CHECK-NEXT: 00000008  00000201 R_PPC_ADDR32           00000000   .debug_abbrev + 0
# CHECK-NEXT: 00000011  00000301 R_PPC_ADDR32           00000000   .debug_str_offsets + 8
# CHECK-NEXT: 00000015  00000501 R_PPC_ADDR32           00000000   .debug_line + 0
# CHECK-NEXT: 0000001f  00000401 R_PPC_ADDR32           00000000   .debug_addr + 8
# CHECK-NEXT: 0000002d  0000074e R_PPC_DTPREL32         00000000   my_tls_var + 8000

	.globl	foo                             # -- Begin function foo
	.p2align	2
	.type	foo,@function
foo:                                    # @foo
.Lfunc_begin0:
	.cfi_startproc
# %bb.0:                                # %entry
	stwu 1, -16(1)
	stw 31, 12(1)
	.cfi_def_cfa_offset 16
	.cfi_offset r31, -4
	mr	31, 1
	.cfi_def_cfa_register r31
.Ltmp0:
	addis 3, 2, my_tls_var@tprel@ha
	addi 4, 3, my_tls_var@tprel@l
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
.Ltmp1:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-.Lfunc_begin0
	.cfi_endproc
                                        # -- End function
	.type	my_tls_var,@object              # @my_tls_var
	.section	.tdata,"awT",@progbits
	.globl	my_tls_var
	.p2align	2, 0x0
my_tls_var:
	.long	1                               # 0x1
	.size	my_tls_var, 4

	.type	my_global_var,@object           # @my_global_var
	.data
	.globl	my_global_var
	.p2align	2, 0x0
my_global_var:
	.long	2                               # 0x2
	.size	my_global_var, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	1                               # DWARF Unit Type
	.byte	4                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	1                               # Abbrev [1] 0xc:0x41 DW_TAG_compile_unit
	.byte	0                               # DW_AT_producer
	.short	29                              # DW_AT_language
	.byte	1                               # DW_AT_name
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.byte	2                               # DW_AT_comp_dir
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	2                               # Abbrev [2] 0x23:0xf DW_TAG_variable
	.byte	3                               # DW_AT_name
	.long	50                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	6                               # DW_AT_location
	.byte	12
	.long	my_tls_var@DTPREL+32768
	.byte	224
	.byte	3                               # Abbrev [3] 0x32:0x4 DW_TAG_base_type
	.byte	4                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x36:0xb DW_TAG_variable
	.byte	5                               # DW_AT_name
	.long	50                              # DW_AT_type
                                        # DW_AT_external
	.byte	0                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	161
	.byte	0
	.byte	4                               # Abbrev [4] 0x41:0xb DW_TAG_subprogram
	.byte	1                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	111
	.byte	6                               # DW_AT_name
	.byte	0                               # DW_AT_decl_file
	.byte	4                               # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	32                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	4                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.long	my_global_var
	.long	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git a4f4d82c35b80b681687b545200456e79a82d9c2)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym my_global_var
	.section	.debug_line,"",@progbits
.Lline_table_start0:
