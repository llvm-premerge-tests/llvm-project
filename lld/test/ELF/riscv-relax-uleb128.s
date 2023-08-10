# REQUIRES: riscv
## Relax R_RISCV_SET_ULEB128 and R_RISCV_SUB_ULEB128.

# RUN: rm -rf %t && split-file %s %t && cd %t

## RELAX
# RUN: llvm-mc -filetype=obj -triple=riscv32 -riscv-enable-uleb128 -mattr=+c,+relax a.s -o a.32c.o
# RUN: ld.lld -T lds a.32c.o -o 32c
# RUN: llvm-objdump -td -j .text -j .data -M no-aliases 32c | FileCheck %s --check-prefixes=RELAX,RELAX32

# RUN: llvm-mc -filetype=obj -triple=riscv64 -riscv-enable-uleb128 -mattr=+c,+relax a.s -o a.64c.o
# RUN: ld.lld -T lds a.64c.o -o 64c
# RUN: llvm-objdump -td -j .text -j .data -M no-aliases 64c | FileCheck %s --check-prefixes=RELAX,RELAX64

## --no-relax disables relaxation.
# RUN: ld.lld -T lds a.64c.o --no-relax -o 64c.norelax
# RUN: llvm-objdump -td -j .text -j .data -M no-aliases 64c.norelax | FileCheck %s --check-prefixes=NORELAX

# RELAX32:       00010002 l       .text  00000000 a
# RELAX32:       00010002 l       .text  00000000 .Lend_start
# RELAX32:       00010000 g       .text  00000002 _start

# RELAX64:       0000000000010004 l       .text  0000000000000000 a
# RELAX64:       0000000000010004 l       .text  0000000000000000 .Lend_start
# RELAX64:       0000000000010000 g       .text  0000000000000004 _start

# RELAX-LABEL:   <_start>:
# RELAX32-NEXT:    10000: 09 20         c.jal   0x10002 <a>
# RELAX64-NEXT:    10000: ef 00 40 00   jal     ra, 0x10004 <a>
# RELAX-EMPTY:
# RELAX-NEXT:    <a>:
# RELAX32-NEXT:      10002: 82 80         c.jr    ra
# RELAX64-NEXT:      10004: 82 80         c.jr    ra
# RELAX-EMPTY:

# RELAX-LABEL:   <.data>:
# RELAX32:        20000: 82 80
# RELAX32-NEXT:   20002: 80 80
# RELAX32-NEXT:   20004: 00
# RELAX64:        20000: 84 80
# RELAX64-NEXT:   20002: 80 80
# RELAX64-NEXT:   20004: 80 80
# RELAX64-NEXT:   20006: 80 80
# RELAX64-NEXT:   20008: 80 00
# RELAX-EMPTY:

# NORELAX-LABEL: <_start>:
# NORELAX:           10000:  97 00 00 00   auipc   ra, 0
# NORELAX-NEXT:              e7 80 80 00   jalr    ra, 8(ra)
# NORELAX-EMPTY:

# NORELAX-LABEL:   <.data>:
# NORELAX:           20000: 88 80
# NORELAX-NEXT:      20002: 80 80
# NORELAX-NEXT:      20004: 80 80
# NORELAX-NEXT:      20006: 80 80
# NORELAX-NEXT:      20008: 80 00
# NORELAX-EMPTY:


#--- a.s
.global _start
_start:
  call a          # rv32c: c.jal; rv64c: jal
.Lend_start:
.size _start, . - _start

a:
  ret
.data
.uleb128 .Lend_start-_start

#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .data 0x20000 : { *(.data) }
}
