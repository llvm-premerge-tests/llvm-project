# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: ld.lld -T %t/lds %t/a.o -o %t/bin
# RUN: llvm-objdump --no-print-imm-hex -d %t/bin | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readobj --sections %t/bin | FileCheck --check-prefixes=SECTIONS,GOT %s
# RUN: ld.lld %t/a.o -o %t/bin2
# RUN: llvm-readobj --sections %t/bin2 | FileCheck --check-prefixes=SECTIONS,UNNECESSARY-GOT %s

# DISASM:      <_start>:
# DISASM-NEXT: movl    26(%rip), %eax
# DISASM-NEXT: movq    19(%rip), %rax
# DISASM-NEXT: leaq    2147483641(%rip), %rax
# DISASM-NEXT: leal    2147483635(%rip), %eax

# Make sure .got still exists with the right size
# SECTIONS:             Name: .got
# SECTIONS-NEXT:        Type: SHT_PROGBITS
# SECTIONS-NEXT:        Flags [
# SECTIONS-NEXT:          SHF_ALLOC
# SECTIONS-NEXT:          SHF_WRITE
# SECTIONS-NEXT:        ]
# SECTIONS-NEXT:        Address:
# SECTIONS-NEXT:        Offset:
# GOT-NEXT:             Size: 8
# UNNECESSARY-GOT-NEXT: Size: 0

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
