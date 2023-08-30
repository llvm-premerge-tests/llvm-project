// RUN: llvm-mc -triple arm64-apple-macosx %s -filetype=obj -o %t
// RUN: llvm-objdump --color --disassemble %t | FileCheck %s --check-prefix=COLOR
// RUN: llvm-objdump --no-color --disassemble %t | FileCheck %s --check-prefix=NOCOLOR

sub	sp, sp, #16
str	w0, [sp, #12]
ldr	w8, [sp, #12]
ldr	w9, [sp, #12]
mul	w0, w8, w9
add	sp, sp, #16


// COLOR: sub	[0;36msp, [0;36msp, [0;31m#0x10
// COLOR: str	[0;36mw0, [[0;36msp, [0;31m#0xc]
// COLOR: ldr	[0;36mw8, [[0;36msp, [0;31m#0xc]
// COLOR: ldr	[0;36mw9, [[0;36msp, [0;31m#0xc]
// COLOR: mul	[0;36mw0, [0;36mw8, [0;36mw9
// COLOR: add	[0;36msp, [0;36msp, [0;31m#0x10

// NOCOLOR: sub	sp, sp, #0x10
// NOCOLOR: str	w0, [sp, #0xc]
// NOCOLOR: ldr	w8, [sp, #0xc]
// NOCOLOR: ldr	w9, [sp, #0xc]
// NOCOLOR: mul	w0, w8, w9
// NOCOLOR: add	sp, sp, #0x10
