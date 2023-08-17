# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds %t/a.o -o /dev/null --print-relocation-stats=%t/stats
# RUN: FileCheck %s --input-file %t/stats

# CHECK: Min 32-bit offset: -99999
# CHECK: Max 32-bit offset: 199994

#--- a.s
.section data1,"aw",@progbits
.space 8

.section .text, "ax", @progbits
  movq __stop_data1@GOTPCREL(%rip), %rax
  movq __stop_data2@GOTPCREL(%rip), %rax

.section data2,"aw",@progbits
.space 8

#--- lds
SECTIONS {
  data1 100000 : { *(data1) }
  .text 200000 : { *(.text) }
  data2 400000 : { *(data2) }
}

