# This test case reproduces a bug where, during register swapping,
# the code fragments associated with the function need to be swapped
# together (which may be generated during PGO optimization). If not
# handled properly, optimized binary execution can result in a segmentation fault.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: %clang -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.out -data=%t.fdata --reg-reassign | FileCheck %s
# RUN: %t.out

# CHECK: BOLT-INFO: Reg Reassignment Pass Stats
# CHECK-NEXT: 2 functions affected.
  .text
  .globl  main                          # -- Begin function main
  .p2align  4, 0x90
  .type   main,@function
  .type   main.cold,@function
main.cold:
.L1:
  mov     $0x3,  %rbx
  jmp     .L3
main:                                   # @main
  .cfi_startproc
# %bb.0:                                # %entry
  pushq   %rax
  pushq   %r12
  pushq   %rbx
  .cfi_def_cfa_offset 16
  mov     $0x1,  %r12
  mov     $0x2,  %rbx
.L2:
  add     $0x1,  %r12
  shr     $0x14, %r12
  jmp     .L1
.L3:
  cmp     $0x3,  %r12
  je      0x0
  xorl    %eax, %eax
  popq    %rcx
  popq    %rbx
  popq    %r12
  .cfi_def_cfa_offset 8
  retq
# FDATA: 1 main.cold/1 7 1 main 0 0 100
# FDATA: 1 main 20 1 main 26 0 100
# FDATA: 1 main 1a 1 main.cold/1 0 0 100

.Lfunc_end0:
  .size  main, .Lfunc_end0-main
  .cfi_endproc
