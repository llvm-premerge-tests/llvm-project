// RUN: llvm-mc -triple=armv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
// RUN:               -abs target=0x76bbe880 -show-entry-es -check %s %t.o | FileCheck --check-prefix=CHECK-ARM %s

// RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
// RUN:               -abs target=0x76bbe880 -show-entry-es -check %s %t.o | FileCheck --check-prefix=CHECK-THM %s

// Check that ARM symbol addresses are aligned to 4 bytes
// CHECK-ARM: "abs32": 0x{{[0-9a-f]+[048c]}} [Data] Ready
// CHECK-ARM: "rel32": 0x{{[0-9a-f]+[048c]}} [Data] Ready

// Check that Thumb symbol addresses are aligned to 2 bytes and Thumb functions have LSB set
// CHECK-THM: "abs32": 0x{{[0-9a-f]+[02468ace]}} [Data] Ready
// CHECK-THM: "rel32": 0x{{[0-9a-f]+[02468ace]}} [Data] Ready

// Check R_ARM_ABS32 and R_ARM_REL32 relocation handling

        .globl target
        .type target, %function
        .globl  main
        .type main, %function
        .p2align  2

main:
        bx lr
        .size   main, .-main

// R_ARM_ABS32
        .global abs32
// jitlink-check: *{4}(abs32) = target
abs32:
        .word target
        .size   abs32, .-abs32

// R_ARM_REL32
        .global rel32
// jitlink-check: (rel32 + *{4}(rel32))[31:0] = target
 rel32:
        .word target - .
        .size rel32, .-rel32

