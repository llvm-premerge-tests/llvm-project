# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds %t/a.o -o %t/bin
# RUN: llvm-objdump --no-print-imm-hex -d %t/bin | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readelf -S %t/bin | FileCheck --check-prefixes=GOT %s
# RUN: ld.lld %t/a.o -o %t/bin2
# RUN: llvm-readelf -S %t/bin2 | FileCheck --check-prefixes=UNNECESSARY-GOT %s

# DISASM:      <_start>:
# DISASM-NEXT: movl    26(%rip), %eax
# DISASM-NEXT: movq    19(%rip), %rax
# DISASM-NEXT: leaq    2147483641(%rip), %rax
# DISASM-NEXT: leal    2147483635(%rip), %eax

# In our implementation, .got is retained even if all GOT-generating relocations are optimized.
# Make sure .got still exists with the right size.
# UNNECESSARY-GOT: .got PROGBITS 00000000002021e8 0001e8 000000 00 WA 0 0 8
#             GOT: .got PROGBITS 0000000000200020 001020 000008 00 WA 0 0 8

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
