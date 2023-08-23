## Test llvm-mc could handle .attribute correctly.

# RUN: llvm-mc %s -triple=riscv32 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm | FileCheck %s
# RUN: llvm-mc %s -triple=riscv32 -filetype=asm -riscv-add-build-attributes \
# RUN:   | FileCheck %s
# RUN: llvm-mc %s -triple=riscv64 -filetype=asm -riscv-add-build-attributes \
# RUN:   | FileCheck %s

.attribute stack_align, 16
# CHECK: attribute      4, 16

.attribute arch, "rv32i2p1_m2p0_a2p1_c2p0"
# CHECK: attribute      5, "rv32i2p1_m2p0_a2p1_c2p0"

.attribute unaligned_access, 0
# CHECK: attribute      6, 0
