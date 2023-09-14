# RUN: llvm-mc -triple=armv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
# RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t.o | FileCheck --check-prefixes=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:               -abs target=0x76bbe880 -show-entry-es -check %s %t.o | FileCheck --check-prefix=CHECK-JIT-ARM %s

# RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -arm-add-build-attributes -filetype=obj -o %t.o %s
# RUN: llvm-objdump -r %t.o | FileCheck --check-prefix=CHECK-TYPE %s
# RUN: llvm-objdump --disassemble %t.o | FileCheck --check-prefixes=CHECK-INSTR %s
# RUN: llvm-jitlink -noexec -slab-address 0x76ff0000 -slab-allocate 10Kb -slab-page-size 4096 \
# RUN:               -abs target=0x76bbe880 -show-entry-es -check %s %t.o | FileCheck --check-prefix=CHECK-JIT-THM %s


	.text
	.syntax unified

	.globl target
	.type target, %function

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_ABS32 target
# CHECK-INSTR: 	00000000 <abs32>:
# CHECK-INSTR: 	       0: 00 00 00 00   .word   0x00000000
# jitlink-check: *{4}(abs32) = target
# Check that ARM symbol addresses are aligned to 4 bytes
# CHECK-JIT-ARM: "abs32": 0x{{[0-9a-f]+[048c]}} [Data] Ready
# Check that Thumb symbol addresses are aligned to 2 bytes
# CHECK-JIT-THM: "abs32": 0x{{[0-9a-f]+[02468ace]}} [Data] Ready
	.global abs32
abs32:
	.word target
	.size   abs32, .-abs32

# CHECK-TYPE: {{[0-9a-f]+}} R_ARM_REL32 target
# CHECK-INSTR: 	00000004 <rel32>:
# CHECK-INSTR: 	       4: 00 00 00 00   .word   0x00000000
# jitlink-check: (rel32 + *{4}(rel32))[31:0] = target
# CHECK-JIT-ARM: "rel32": 0x{{[0-9a-f]+[048c]}} [Data] Ready
# CHECK-JIT-THM: "rel32": 0x{{[0-9a-f]+[02468ace]}} [Data] Ready
	.global rel32
 rel32:
	.word target - .
	.size rel32, .-rel32

	.globl  main
	.type main, %function
	.p2align  2

main:
	bx lr
	.size   main, .-main
