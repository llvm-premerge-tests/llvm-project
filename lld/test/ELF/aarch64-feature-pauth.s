# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag1.s -o tag11.o
# RUN: cp tag11.o tag12.o
# RUN: ld.lld -shared tag11.o tag12.o -o tagok.so
# RUN: llvm-readelf -n tagok.so | FileCheck --check-prefix OK %s

# OK: AArch64 PAuth ABI tag: platform 0x2a, version 0x1

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag2.s -o tag2.o
# RUN: not ld.lld tag11.o tag12.o tag2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR1 %s

# ERR1: error: incompatible values of AArch64 PAuth compatibility info found
# ERR1: {{.*}}: 0x2A000000000000000{{1|2}}00000000000000
# ERR1: {{.*}}: 0x2A000000000000000{{1|2}}00000000000000

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-errs.s -o errs.o
# RUN: not ld.lld errs.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR2 %s

# ERR2:      error: {{.*}}: invalid type field value 42 (1 expected)
# ERR2-NEXT: error: {{.*}}: invalid name field value XXX (ARM expected)
# ERR2-NEXT: error: {{.*}}: too short AArch64 PAuth compatibility info (at least 16 bytes expected)

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag-short.s -o short.o
# RUN: not ld.lld short.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR3 %s

# ERR3: error: {{.*}}: section is too short

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu no-info.s -o noinfo1.o
# RUN: cp noinfo1.o noinfo2.o
# RUN: not ld.lld -z pauth-report=error tag11.o noinfo1.o noinfo2.o -o /dev/null 2>&1 | FileCheck --check-prefix ERR4 %s
# RUN: ld.lld -z pauth-report=warning tag11.o noinfo1.o noinfo2.o -o /dev/null 2>&1 | FileCheck --check-prefix WARN %s
# RUN: ld.lld -z pauth-report=none tag11.o noinfo1.o noinfo2.o -o /dev/null 2>&1 | FileCheck --check-prefix NONE %s

# ERR4:      error: {{.*}}noinfo1.o has no AArch64 PAuth compatibility info while {{.*}}tag11.o has one; either all or no input files must have it
# ERR4-NEXT: error: {{.*}}noinfo2.o has no AArch64 PAuth compatibility info while {{.*}}tag11.o has one; either all or no input files must have it
# WARN:      warning: {{.*}}noinfo1.o has no AArch64 PAuth compatibility info while {{.*}}tag11.o has one; either all or no input files must have it
# WARN-NEXT: warning: {{.*}}noinfo2.o has no AArch64 PAuth compatibility info while {{.*}}tag11.o has one; either all or no input files must have it
# NONE-NOT:  {{.*}} has no AArch64 PAuth compatibility info while {{.*}} has one; either all or no input files must have it

#--- abi-tag-short.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 8

#--- abi-tag-errs.s

.section ".note.AARCH64-PAUTH-ABI-tag", "a"
.long 4
.long 8
.long 42
.asciz "XXX"

.quad 42

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

#--- no-info.s

.section ".test", "a"
