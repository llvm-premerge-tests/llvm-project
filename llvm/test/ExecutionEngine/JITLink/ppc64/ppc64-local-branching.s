# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=powerpc64le-unknown-linux-gnu --filetype=obj -o \
# RUN:   %t/elf_reloc.o %s
# RUN: not llvm-jitlink --noexec %t/elf_reloc.o 2>&1 | FileCheck %s
# RUN: llvm-mc --triple=powerpc64-unknown-linux-gnu --filetype=obj -o \
# RUN:   %t/elf_reloc.o %s
# RUN: not llvm-jitlink --noexec %t/elf_reloc.o 2>&1 | FileCheck %s
# CHECK: relocation target $__STUBS{{.*}} is out of range of CallBranchDelta fixup at {{.*}}

  .text
  .abiversion 2
  .global main
  .p2align 4
  .type main,@function
main:
  addis 2, 12, .TOC.-main@ha
  addi 2, 2, .TOC.-main@l
  .localentry main, .-main
  bl foo
  li 3, 0
  blr
  .size main, .-main

  .global foo
  .p2align 4
  .type foo,@function
foo:
  addis 2, 12, .TOC.-foo@ha
  addi 2, 2, .TOC.-foo@l
  .localentry foo, .-foo
  li 3, 1
  bl bar
  blr
  .size foo, .-foo

# Skip 32M so that bar is out of range of R_PPC64_REL24 in foo.
  .skip 33554432 , 0

  .global bar
  .p2align 4
  .type bar,@function
bar:
  addis 2, 12, .TOC.-bar@ha
  addi 2, 2, .TOC.-bar@l
  .localentry bar, .-bar
  li 3, 2
  blr
  .size bar, .-bar

