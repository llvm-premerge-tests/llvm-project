// RUN: split-file %s %t

//--- ok.s

// RUN: llvm-mc -triple=aarch64 %t/ok.s | \
// RUN: FileCheck %s --check-prefix=ASM

// RUN: llvm-mc -triple=aarch64 -filetype=obj %t/ok.s | \
// RUN: llvm-readelf -S -r -x .test - | \
// RUN: FileCheck %s --check-prefix=RELOC

// RELOC: Relocation section '.rela.test' at offset 0x1e8 contains 8 entries:
// RELOC-NEXT:  Offset Info Type Symbol's Value Symbol's Name + Addend
// RELOC-NEXT: 0000000000000000 000000010000e100 R_AARCH64_AUTH_ABS64 0000000000000000 .helper + 0
// RELOC-NEXT: 0000000000000010 000000060000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g1 + 0
// RELOC-NEXT: 0000000000000020 000000070000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g2 + 0
// RELOC-NEXT: 0000000000000030 000000080000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g3 + 0
// RELOC-NEXT: 0000000000000040 000000090000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g4 + 7
// RELOC-NEXT: 0000000000000050 0000000a0000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g5 - 3
// RELOC-NEXT: 0000000000000060 000000020000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g 6 + 0
// RELOC-NEXT: 0000000000000070 0000000b0000e100 R_AARCH64_AUTH_ABS64 0000000000000000 _g 7 + 7

// RELOC: Hex dump of section '.test':
//                VVVVVVVV addend, not needed for rela
//                             VV reserved
// RELOC-NEXT: 00 00000000 2a000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT: 10 00000000 00000010
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT: 20 00000000 050000a0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 10 da key 0000 reserved
// RELOC-NEXT: 30 00000000 ffff00b0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT: 40 00000000 00000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT: 50 00000000 00de0010
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT: 60 00000000 ff0000b0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT: 70 00000000 10000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved

.section    .helper
.local "_g 6"
.type _g0, @function
_g0:
  ret

.section	.test
.p2align	3

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

//--- err1.s

// RUN: not llvm-mc -triple=aarch64 %t/err1.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR1

// ERR1: error: expected '('

.quad sym@AUTH)ia,42)

//--- err2.s

// RUN: not llvm-mc -triple=aarch64 %t/err2.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR2

// ERR2: error: expected key name

.quad sym@AUTH(42,42)

//--- err3.s

// RUN: not llvm-mc -triple=aarch64 %t/err3.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR3

// ERR3: error: invalid key 'ic'

.quad sym@AUTH(ic,42)

//--- err4.s

// RUN: not llvm-mc -triple=aarch64 %t/err4.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR4

// ERR4: error: expected ','

.quad sym@AUTH(ia 42)

//--- err5.s

// RUN: not llvm-mc -triple=aarch64 %t/err5.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR5

// ERR5: error: expected integer discriminator

.quad sym@AUTH(ia,xxx)

//--- err6.s

// RUN: not llvm-mc -triple=aarch64 %t/err6.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR6

// ERR6: error: integer discriminator 65536 is out of range. Permitted range is [0, 0xFFFF]

.quad sym@AUTH(ia,65536)

//--- err7.s

// RUN: not llvm-mc -triple=aarch64 %t/err7.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR7

// ERR7: error: expected 'addr'

.quad sym@AUTH(ia,42,add)

//--- err8.s

// RUN: not llvm-mc -triple=aarch64 %t/err8.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR8

// ERR8: error: expected ')'

.quad sym@AUTH(ia,42(

//--- err9.s

// RUN: not llvm-mc -triple=aarch64 %t/err9.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR9

// ERR9: error: combination of @AUTH with other modifiers not supported

.quad sym@PLT@AUTH(ia,42)

//--- err10.s

// RUN: not llvm-mc -triple=aarch64 %t/err10.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR10

// ERR10: error: invalid variant 'AUTH@GOT'

.quad sym@AUTH@GOT(ia,42)

//--- err11.s

// RUN: not llvm-mc -triple=aarch64 %t/err11.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR11

// ERR11: error: invalid variant 'TLSDESC@AUTH'

.quad "long sym"@TLSDESC@AUTH(ia,42)

//--- err12.s

// RUN: not llvm-mc -triple=aarch64 %t/err12.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR12

// ERR12: error: invalid variant 'AUTH@PLT'

.quad "long sym"@AUTH@PLT(ia,42)

//--- err13.s

// RUN: not llvm-mc -triple=aarch64 %t/err13.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR13

// ERR13: error: invalid variant 'GOT@AUTH'

.quad (sym - 5)@GOT@AUTH(ia,42)

//--- err14.s

// RUN: not llvm-mc -triple=aarch64 %t/err14.s 2>&1 | \
// RUN: FileCheck %s --check-prefix=ERR14

// ERR14: error: invalid variant 'AUTH@TLSDESC'

.quad (sym + 5)@AUTH@TLSDESC(ia,42)