//===-- xray_trampoline_riscv32.s ----------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the riscv32-specific assembler for the trampolines.
//
//===----------------------------------------------------------------------===//

#include "../builtins/assembly.h"
#include "../sanitizer_common/sanitizer_asm.h"

	.text
	.p2align 2
	.global ASM_SYMBOL(__xray_FunctionEntry)
	ASM_TYPE_FUNCTION(__xray_FunctionEntry)
	ASM_SYMBOL(__xray_FunctionEntry):
	.cfi_startproc
	// Push argument registers to stack
	addi	sp, sp, -100
	.cfi_def_cfa_offset 100
	sw	ra, 96(sp)
	.cfi_offset ra, -8
	sw	a7, 92(sp)
	sw	a6, 88(sp)
	sw	a5, 84(sp)
	sw	a4, 80(sp)
	sw	a3, 76(sp)
	sw	a2, 72(sp)
	sw	a1, 68(sp)
	sw	a0, 64(sp)
	// The current implementation only supports double precision floats
	// In case of RISCV32F, then these (faX) would still be 32 bit
	// wide registers, so we should be incrementing by 4
	fsd	fa7, 56(sp)
	fsd	fa6, 48(sp)
	fsd	fa5, 40(sp)
	fsd	fa4, 32(sp)
	fsd	fa3, 24(sp)
	fsd	fa2, 16(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)

	// Load the handler function pointer into a2
	li	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
	lw	a2, 0(a2)

	// Handler address will be null if it is not set
	beq	a2, x0, FunctionEntry_restore

	// If we reach here, we are tracing an event
	// a0 already contains function id
	// a1 = 0 means we are tracing an entry event
	mv	a1, x0
	jalr	a2

FunctionEntry_restore:
	// Restore argument registers
	fld	fa0, 0(sp)
	fld	fa1, 8(sp)
	fld	fa2, 16(sp)
	fld	fa3, 24(sp)
	fld	fa4, 32(sp)
	fld	fa5, 40(sp)
	fld	fa6, 48(sp)
	fld	fa7, 56(sp)
	lw	a0, 64(sp)
	lw	a1, 68(sp)
	lw	a2, 72(sp)
	lw	a3, 76(sp)
	lw	a4, 80(sp)
	lw	a5, 84(sp)
	lw	a6, 88(sp)
	lw	a7, 92(sp)
	lw	ra, 96(sp)
	addi	sp, sp, 100
	jr	ra

FunctionEntry_end:
	ASM_SIZE(__xray_FunctionEntry)
	.cfi_endproc

	.text
	.p2align 2
	.global ASM_SYMBOL(__xray_FunctionExit)
	ASM_TYPE_FUNCTION(__xray_FunctionExit)
	ASM_SYMBOL(__xray_FunctionExit):
	.cfi_startproc
	// Push return registers to stack
	addi	sp, sp, -28
	.cfi_def_cfa_offset 28
	sw	ra, 24(sp)
	.cfi_offset ra, -8
	sw	a1, 20(sp)
	sw	a0, 16(sp)
	// The current implementation only supports double precision floats
	// In case of RISCV32F, then these (faX) would still be 32 bit
	// wide registers, so we should be incrementing by 4
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)

	// Load the handler function pointer into a2
	li	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
	lw	a2, 0(a2)

	// Handler address will be null if it is not set
	beq	a2, x0, FunctionExit_restore

	// If we reach here, we are tracing an event
	// a0 already contains function id
	// a1 = 1 means we are tracing an exit event
	addi	a1, x0, 1
	jalr	a2

FunctionExit_restore:
	// Restore return registers
	fld	fa0, 0(sp)
	fld	fa1, 8(sp)
	lw	a0, 16(sp)
	lw	a1, 20(sp)
	lw	ra, 24(sp)
	addi	sp, sp, 28
	jr	ra

FunctionExit_end:
	ASM_SIZE(__xray_FunctionExit)
	.cfi_endproc
