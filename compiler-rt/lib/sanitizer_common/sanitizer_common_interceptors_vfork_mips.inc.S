#if defined(__mips) && defined(__linux__)

#if defined(__mips64)
# define ADDI daddiu
# define ADDU DADDU
# define ST sd
# define LD ld
# define LA dla
# define GOT_DISP got_disp
#else
# define ADDI addiu
# define ADDU addu
# define ST sw
# define LD lw
# define LA la
# define GOT_DISP got
#endif

// MEMOEY USAGE:
// SP[0]: ra; SP[8]: gp
// SPILL[0]: ra; SPILL[8]: s8

#include "sanitizer_common/sanitizer_asm.h"

ASM_HIDDEN(COMMON_INTERCEPTOR_SPILL_AREA)

.comm _ZN14__interception10real_vforkE,__SIZEOF_POINTER__,__SIZEOF_POINTER__
.globl ASM_WRAPPER_NAME(vfork)
ASM_TYPE_FUNCTION(ASM_WRAPPER_NAME(vfork))
ASM_WRAPPER_NAME(vfork):
        ADDI	$sp, $sp, -16
        ST	$ra, 0($sp)
#if __mips_isa_rev < 6
	// __interceptor_trampoline_vfork call us by `b`, so the t9 contains the
	// address of __interceptor_trampoline_vfork instead of __interceptor_vfork.
#  if defined(__mips64)
	// Compute $gp
	lui	$gp, %hi(%neg(%gp_rel(__interceptor_trampoline_vfork)))
	ADDU	$gp, $gp, $t9
	ADDI	$gp, $gp, %lo(%neg(%gp_rel(__interceptor_trampoline_vfork)))
#  else
	.set	noreorder
	#nal FIXME: nal is not supported by llvm yet. So use bltzal now.
	# It is used to get the value of PC.
	bltzal	$0, .L_exit
	nop
	addiu	$t9, $ra, 4
	.cpload	$t9
	.set	reorder
#  endif
#endif

#if __mips_isa_rev >= 6
        balc	COMMON_INTERCEPTOR_SPILL_AREA
#else
	LD	$t9, %GOT_DISP(COMMON_INTERCEPTOR_SPILL_AREA)($gp)
        ST	$gp, 8($sp)
        jalr	$t9

        LD	$gp, 8($sp)
	// save s8, and move gp to s0, so that we can always known our gp.
	// we didn't it earlier, due to that we need COMMON_INTERCEPTOR_SPILL_AREA,
	// which can keep data cross vfork call.
        ST	$s8, 8($v0)
        MOVE	$s8, $gp
#endif
        // restore previous values from stack
        LD	$ra, 0($sp)
        // adjust stack
        ADDI	$sp, $sp, 16
        // Save ra in the off-stack spill area.
        ST	$ra, 0($v0)

        // Call real vfork. This may return twice. User code that runs between the first and the second return
        // may clobber the stack frame of the interceptor; that's why it does not have a frame.
	// do *NOT* use s8, since it is not set by MIPSr6.
	LD	$t9, %GOT_DISP(_ZN14__interception10real_vforkE)($gp)
	LD	$t9, 0($t9)
        jalr	$t9

        // adjust stack
        ADDI	$sp, $sp, -16
        // store v0 by adjusted stack
        sd	$v0, 8($sp)
        // jump to exit label if v0 is 0 => child process
        beqz	$v0, .L_exit

        // v0 != 0 => parent process. Clear stack shadow.
        ADDI	$a0, $sp, 16
#if __mips_isa_rev >= 6
        balc	COMMON_INTERCEPTOR_HANDLE_VFORK
#else
	// here, s8 has the value of gp
        LD	$t9, %GOT_DISP(COMMON_INTERCEPTOR_HANDLE_VFORK)($s8)
	jalr	$t9
#endif

.L_exit:
#if __mips_isa_rev >= 6
        balc	COMMON_INTERCEPTOR_SPILL_AREA
#else
	// here, s8 has the value of gp
	LD	$t9, %GOT_DISP(COMMON_INTERCEPTOR_SPILL_AREA)($s8)
        jalr	$t9

        LD	$s8, 8($v0)
#endif
        // Restore ra
        LD	$ra, 0($v0)
        // the return value of vfork
        LD	$v0, 8($sp)
        ADDI	$sp, $sp, 16
        jr	$ra
ASM_SIZE(vfork)

ASM_INTERCEPTOR_TRAMPOLINE(vfork)
ASM_TRAMPOLINE_ALIAS(vfork, vfork)

#endif
