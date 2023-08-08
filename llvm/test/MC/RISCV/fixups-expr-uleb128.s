# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NO-EMIT-RELOC %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NORELAX,NO-EMIT-RELOC %s

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax \
# RUN:         %s -riscv-enable-uleb128 \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,EMIT-RELOC,EMIT-RELOC-32 %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax \
# RUN:         %s -riscv-enable-uleb128 \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NORELAX,NO-EMIT-RELOC %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NO-EMIT-RELOC %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NORELAX,NO-EMIT-RELOC %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax \
# RUN:         %s -riscv-enable-uleb128 \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,EMIT-RELOC,EMIT-RELOC-64 %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax \
# RUN:         %s -riscv-enable-uleb128 \
# RUN:     | llvm-readobj -r - \
# RUN:     | FileCheck -check-prefixes CHECK,NORELAX,NO-EMIT-RELOC %s

# Check that subtraction expressions are emitted as two relocations always.

.text
.globl G1
.globl G2
.L1:
G1:
call extern
.L2:
G2:

.data
.uleb128 G3-G1
.uleb128 .L2-.L1
.uleb128 G2-G1

# NO-EMIT-RELOC-NOT: R_RISCV_SET_ULEB128
# NO-EMIT-RELOC-NOT: R_RISCV_SUB_ULEB128

# CHECK: Relocations [

# NORELAX-NEXT:   .rela.text {
# NORELAX-NEXT:     R_RISCV_CALL_PLT
# NORELAX-NEXT:   }

# EMIT-RELOC: Section (5) .rela.data {
# EMIT-RELOC-32:      0x0 R_RISCV_SET_ULEB128 G3 0x0
# EMIT-RELOC-32-NEXT: 0x0 R_RISCV_SUB_ULEB128 G1 0x0
# EMIT-RELOC-32-NEXT: 0x5 R_RISCV_SET_ULEB128 .L2 0x0
# EMIT-RELOC-32-NEXT: 0x5 R_RISCV_SUB_ULEB128 .L1 0x0
# EMIT-RELOC-32-NEXT: 0xA R_RISCV_SET_ULEB128 G2 0x0
# EMIT-RELOC-32-NEXT: 0xA R_RISCV_SUB_ULEB128 G1 0x0

# EMIT-RELOC-64:      0x0 R_RISCV_SET_ULEB128 G3 0x0
# EMIT-RELOC-64-NEXT: 0x0 R_RISCV_SUB_ULEB128 G1 0x0
# EMIT-RELOC-64-NEXT: 0xA R_RISCV_SET_ULEB128 .L2 0x0
# EMIT-RELOC-64-NEXT: 0xA R_RISCV_SUB_ULEB128 .L1 0x0
# EMIT-RELOC-64-NEXT: 0x14 R_RISCV_SET_ULEB128 G2 0x0
# EMIT-RELOC-64-NEXT: 0x14 R_RISCV_SUB_ULEB128 G1 0x0
# EMIT-RELOC: }
# CHECK: ]

.text
.globl G3
G3:
nop
