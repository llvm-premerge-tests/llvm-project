# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds %t/a.o -o %t/bin
# RUN: llvm-objdump --no-print-imm-hex -d %t/bin | FileCheck --check-prefix=DISASM %s

# DISASM:      <_start>:
# DISASM-NEXT: movl    -2097158(%rip), %eax
# DISASM-NEXT: movq    -2097165(%rip), %rax
# DISASM-NEXT: leaq    2147483641(%rip), %rax
# DISASM-NEXT: leal    2147483635(%rip), %eax

#--- a.s
.globl _start
.type _start, @function
_start:
  movl __stop_data@GOTPCREL(%rip), %eax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # out of range
  movq __stop_data@GOTPCREL(%rip), %rax  # in range
  movl __stop_data@GOTPCREL(%rip), %eax  # in range

.section data,"aw",@progbits
.space 13

#--- lds
SECTIONS {
  .text 0x200000 : { *(.text) }
  data 0x80200000 : { *(data) }
}
