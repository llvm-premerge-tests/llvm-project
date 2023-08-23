# Require: aarch64

# RUN: split-file %s %tsplit

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/gnu-note1.s -o %tgnu11.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/gnu-note1.s -o %tgnu12.o
# RUN: ld.lld %tgnu11.o %tgnu12.o -o %tgnuok.so
# RUN: llvm-readelf -n %tgnuok.so | FileCheck --check-prefix OK1 %s

# OK1: Properties:    aarch64 feature PAUTH: platform 0x2a, version 0x1

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag1.s -o %ttag11.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag1.s -o %ttag12.o
# RUN: ld.lld %ttag11.o %ttag12.o -o %ttagok.so
# RUN: llvm-readelf -n %ttagok.so | FileCheck --check-prefix OK2 -dump-input=always %s

# OK2: aarch64 PAUTH ABI tag: platform 0x2a, version 0x1

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/gnu-note2.s -o %tgnu2.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag2.s -o %ttag2.o
# RUN: not ld.lld %tgnu11.o %tgnu12.o %tgnu2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR1 %s
# RUN: not ld.lld %ttag11.o %ttag12.o %ttag2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR1 %s

# ERR1: ld.lld: error: Incompatible values of aarch64 pauth compatibility info found
# ERR1: {{.*}}: 0x2A000000000000000{{1|2}}00000000000000
# ERR1: {{.*}}: 0x2A000000000000000{{1|2}}00000000000000

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/gnu-note-invalid-length.s -o %tgnulen.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag-invalid-length.s -o %ttaglen.o
# RUN: not ld.lld %tgnulen.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR2 %s
# RUN: not ld.lld %ttaglen.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR2 %s

# ERR2: ld.lld: error: {{.*}}: too short aarch64 pauth compatibility info (at least 16 bytes expected)

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag-invalid-name.s -o %ttagname.o
# RUN: not ld.lld %ttagname.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR3 %s

# ERR3: ld.lld: error: {{.*}}: invalid name field value XXX (ARM expected)

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/abi-tag-invalid-type.s -o %ttagtype.o
# RUN: not ld.lld %ttagtype.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR4 %s

# ERR4: ld.lld: error: {{.*}}: invalid type field value 42 (1 expected)

# RUN: cat %tsplit/abi-tag1.s %tsplit/gnu-note1.s > %tboth1.s
# RUN: cat %tsplit/abi-tag1.s %tsplit/gnu-note1.s > %tboth2.s
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tboth1.s -o %tboth1.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tboth2.s -o %tboth2.o
# RUN: not ld.lld %tboth1.o %tboth2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR5 %s

# ERR5: ld.lld: error: Input files contain both a .note.AARCH64-PAUTH-ABI-tag section and a GNU_PROPERTY_AARCH64_FEATURE_PAUTH features in a .note.gnu.property section. All the link units must use the same way of specifying aarch64 pauth compatibility info.

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/missing-in-gnu-note.s -o %tgnumiss.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %tsplit/no-info.s -o %tnoinfo.o
# RUN: not ld.lld %tgnu11.o %tgnumiss.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR6 %s
# RUN: not ld.lld %tgnu11.o %tnoinfo.o  -o /dev/null 2>&1 | FileCheck --check-prefix ERR6 %s
# RUN: not ld.lld %ttag11.o %tnoinfo.o  -o /dev/null 2>&1 | FileCheck --check-prefix ERR6 %s

# ERR6: ld.lld: error: {{.*}} has no aarch64 pauth compatibility info while {{.*}} has one. Either all or no link units must have it.

#--- abi-tag-invalid-length.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 8
.long 1
.asciz "ARM"

.quad 42

#--- abi-tag-invalid-name.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "XXX"

.quad 42         // platform
.quad 1          // version

#--- abi-tag-invalid-type.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 42
.asciz "ARM"

.quad 42         // platform
.quad 1          // version

#--- abi-tag1.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 1          // version

#--- abi-tag2.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 16
.long 1
.asciz "ARM"

.quad 42         // platform
.quad 2          // version

#--- gnu-note-invalid-length.s

.section ".note.gnu.property", "a"
.long 4
.long 16
.long 5
.asciz "GNU"

.long 0xc0000001 // GNU_PROPERTY_AARCH64_FEATURE_PAUTH
.long 8
.quad 42

#--- gnu-note1.s

.section ".note.gnu.property", "a"
.long 4
.long 24
.long 5
.asciz "GNU"

.long 0xc0000001 // GNU_PROPERTY_AARCH64_FEATURE_PAUTH
.long 16
.quad 42         // platform
.quad 1          // version

#--- gnu-note2.s

.section ".note.gnu.property", "a"
.long 4
.long 24
.long 5
.asciz "GNU"

.long 0xc0000001 // GNU_PROPERTY_AARCH64_FEATURE_PAUTH
.long 16
.quad 42         // platform
.quad 2          // version

#--- missing-in-gnu-note.s

.section ".note.gnu.property", "a"
.long 4
.long 16
.long 5
.asciz "GNU"

.long 0x42424242 // some dummy type
.long 8
.long 0
.long 0

#--- no-info.s

.section ".test", "a"
