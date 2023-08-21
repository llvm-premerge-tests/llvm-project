//===-- xray_trampoline_riscv64.s ----------------------------------*- ASM -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the riscv64-specific assembler for the trampolines.
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
	addi	sp, sp, -136
	.cfi_def_cfa_offset 136
	sd	ra, 128(sp)
	.cfi_offset ra, -8
	sd	a7, 120(sp)
	sd	a6, 112(sp)
	sd	a5, 104(sp)
	sd	a4, 96(sp)
	sd	a3, 88(sp)
	sd	a2, 80(sp)
	sd	a1, 72(sp)
	sd	a0, 64(sp)
	fsd	fa7, 56(sp)
	fsd	fa6, 48(sp)
	fsd	fa5, 40(sp)
	fsd	fa4, 32(sp)
	fsd	fa3, 24(sp)
	fsd	fa2, 16(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)

	// Load the handler function pointer into a2
	la	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
	ld	a2, 0(a2)

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
	ld	a0, 64(sp)
	ld	a1, 72(sp)
	ld	a2, 80(sp)
	ld	a3, 88(sp)
	ld	a4, 96(sp)
	ld	a5, 104(sp)
	ld	a6, 112(sp)
	ld	a7, 120(sp)
	ld	ra, 128(sp)
	addi	sp, sp, 136
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
	addi	sp, sp, -40
	.cfi_def_cfa_offset 40
	sd	ra, 32(sp)
	.cfi_offset ra, -8
	sd	a1, 24(sp)
	sd	a0, 16(sp)
	fsd	fa1, 8(sp)
	fsd	fa0, 0(sp)

	// Load the handler function pointer into a2
	la	a2, ASM_SYMBOL(_ZN6__xray19XRayPatchedFunctionE)
	ld	a2, 0(a2)

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
	ld	a0, 16(sp)
	ld	a1, 24(sp)
	ld	ra, 32(sp)
	addi	sp, sp, 40
	jr	ra

FunctionExit_end:
	ASM_SIZE(__xray_FunctionExit)
	.cfi_endproc
