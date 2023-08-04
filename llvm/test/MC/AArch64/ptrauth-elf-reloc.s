// RUN: llvm-mc -triple=aarch64 %s | \
// RUN: FileCheck %s --check-prefix=ASM

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | \
// RUN: llvm-readobj --expand-relocs --sections --section-relocations --section-data - | \
// RUN: FileCheck %s --check-prefix=RELOC


// RELOC:    Sections [
// RELOC-LABEL: Section {
// RELOC-LABEL: Section {
// RELOC-LABEL: Section {
// RELOC-LABEL: Section {

// Check signing schema encoding in place of relocation to be applied

// RELOC-NEXT:   Index: 3
// RELOC-NEXT:   Name: .test
// RELOC-LABEL: SectionData (
//                     VVVVVVVV addend, not needed for rela
//                                  VV reserved
// RELOC-NEXT:   0000: 00000000 2A000000
//                              ^^^^ discriminator
//                                    ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT:   0010: 00000000 00000010
//                              ^^^^ discriminator
//                                    ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT:   0020: 00000000 050000A0
//                              ^^^^ discriminator
//                                    ^^ 1    addr diversity 0 reserved 10 da key 0000 reserved
// RELOC-NEXT:   0030: 00000000 FFFF00B0
//                              ^^^^ discriminator
//                                    ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT:   0040: 00000000 00000000
//                              ^^^^ discriminator
//                                    ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT:   0050: 00000000 00DE0010
//                              ^^^^ discriminator
//                                    ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT:   0060: 00000000 FF0000B0
//                              ^^^^ discriminator
//                                    ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT:   0070: 00000000 10000000
//                              ^^^^ discriminator
//                                    ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-LABEL: Section {

// Check relocations themselves

// RELOC-NEXT:   Index: 4
// RELOC-NEXT:   Name: .rela.test

.section	.test
.p2align	3

// RELOC-LABEL: Relocations [
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x0
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g0
// RELOC-NEXT:     Addend: 0x0
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x10
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g1
// RELOC-NEXT:     Addend: 0x0
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x20
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g2
// RELOC-NEXT:     Addend: 0x0
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x30
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g3
// RELOC-NEXT:     Addend: 0x0
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x40
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g4
// RELOC-NEXT:     Addend: 0x7
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x50
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g5
// RELOC-NEXT:     Addend: 0xFFFFFFFFFFFFFFFD
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x60
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g 6
// RELOC-NEXT:     Addend: 0x0
// RELOC-NEXT:   }
// RELOC-NEXT:   Relocation {
// RELOC-NEXT:     Offset: 0x70
// RELOC-NEXT:     Type: R_AARCH64_AUTH_ABS64 (57600)
// RELOC-NEXT:     Symbol: _g 7
// RELOC-NEXT:     Addend: 0x7
// RELOC-NEXT:   }
// RELOC-NEXT: ]

// ASM:          .xword _g0@AUTH(ia,42)
.quad _g0@AUTH(ia,42)
.quad 0

// ASM:          .xword _g1@AUTH(ib,0)
.quad _g1@AUTH(ib,0)
.quad 0

// ASM:          .xword _g2@AUTH(da,5,addr)
.quad _g2@AUTH(da,5,addr)
.quad 0

// ASM:          .xword _g3@AUTH(db,65535,addr)
.quad _g3@AUTH(db,0xffff,addr)
.quad 0

// ASM:          .xword (_g4+7)@AUTH(ia,0)
.quad (_g4 + 7)@AUTH(ia,0)
.quad 0

// ASM:          .xword (_g5-3)@AUTH(ib,56832)
.quad (_g5 - 3)@AUTH(ib,0xde00)
.quad 0

// ASM:          .xword "_g 6"@AUTH(db,255,addr)
.quad "_g 6"@AUTH(db,0xff,addr)
.quad 0

// ASM:          .xword ("_g 7"+7)@AUTH(ia,16)
.quad ("_g 7" + 7)@AUTH(ia,16)
.quad 0
