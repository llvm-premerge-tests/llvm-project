#if defined(__mips) && defined(__linux__)

#if defined(__mips64)
# define ADDI daddiu
# define ST sd
# define LD ld
# define LA dla
#else
# define ADDI addiu
# define ST sw
# define LD lw
# define LA la
#endif

#include "sanitizer_common/sanitizer_asm.h"

ASM_HIDDEN(COMMON_INTERCEPTOR_SPILL_AREA)

.comm _ZN14__interception10real_vforkE,__SIZEOF_POINTER__,__SIZEOF_POINTER__
.globl ASM_WRAPPER_NAME(vfork)
ASM_TYPE_FUNCTION(ASM_WRAPPER_NAME(vfork))
ASM_WRAPPER_NAME(vfork):
        // Save ra in the off-stack spill area.
        // allocate space on stack
        ADDI	$sp, $sp, -16
        // store ra value
        ST	$ra, 8($sp)
#if __mips_isa_rev >= 6
        balc	COMMON_INTERCEPTOR_SPILL_AREA
#else
        LA	$t9, COMMON_INTERCEPTOR_SPILL_AREA
	jalr	$t9
#endif
        // restore previous values from stack
        LD	$ra, 8($sp)
        // adjust stack
        ADDI	$sp, $sp, 16
        // store ra by v0
        ST	$ra, 0($v0)

        // Call real vfork. This may return twice. User code that runs between the first and the second return
        // may clobber the stack frame of the interceptor; that's why it does not have a frame.
        LA	$t9, _ZN14__interception10real_vforkE
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
        LA	$t9, COMMON_INTERCEPTOR_HANDLE_VFORK
	jalr	$t9
#endif

.L_exit:
        // Restore ra
#if __mips_isa_rev >= 6
        balc	COMMON_INTERCEPTOR_SPILL_AREA
#else
        LA	$t9, COMMON_INTERCEPTOR_SPILL_AREA
	jalr	$t9
#endif

        LD	$ra, 8($v0)
        // the return value of vfork
        LD	$v0, 8($sp)
        ADDI	$sp, $sp, 16
        jr	$ra
ASM_SIZE(vfork)

ASM_INTERCEPTOR_TRAMPOLINE(vfork)
ASM_TRAMPOLINE_ALIAS(vfork, vfork)

#endif
