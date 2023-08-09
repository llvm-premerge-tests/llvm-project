// RUN: llvm-mc -triple=armv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
// RUN:               -abs target=0x76bbe880 -show-entry-es -check %s %t.o

# Check R_ARM_CALL relocation handling


        .globl target
        .type target, %function
        .globl  main
        .type main,%function
        .p2align  2

main:
        bx lr
        .size   main, .-main

// R_ARM_CALL
        .global call
# jitlink-check: next_pc(call)+ decode_operand(call, 0) + 4 = target
 call:
        bl target
        .size   call, .-call

// R_ARM_CALL with exchange
        .global callx
# jitlink-check: next_pc(callx)+ decode_operand(callx, 0) + 4 = target
 callx:
        blx target
        .size   callx, .-callx

