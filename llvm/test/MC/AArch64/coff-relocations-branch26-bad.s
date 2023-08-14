// RUN: not llvm-mc -triple aarch64-unknown-windows-msvc -filetype obj %s -o /dev/null 2>&1 | FileCheck %s

    .text
main:
    nop
    b .Ltarget+4

    .def .Ltarget
    .scl 3
    .type 32
    .p2align 2
    .endef
.Ltarget:
    nop
    nop
    ret

// CHECK: error: cannot perform a PC-relative fixup with a non-zero symbol offset
