# RUN: clang --target=mipsel-linux-gnu -march=mips32r6 -c %s -o tmp.o
# RUN: llvm-objdump -d tmp.o | FileCheck %s --check-prefix=MIPSELR6

# MIPSELR6: beqzc	$9, 0x0 <aaa>
# MIPSELR6-NEXT: b	0x0 <aaa>
# MIPSELR6: beqzc	$9, 0x8 <bbb>
# MIPSELR6-NEXT: nop <aaa>
# MIPSELR6: b	0x8 <bbb>
	.set noreorder
aaa:
	beqzc $t1, aaa
	b aaa
	.set reorder
bbb:
	beqzc $t1, bbb
	b bbb
