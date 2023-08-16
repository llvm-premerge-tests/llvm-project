# RUN: not llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+c -riscv-no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s
# RUN: not llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+c -riscv-no-aliases -show-encoding \
# RUN:     2>&1 | FileCheck -check-prefixes=CHECK-ERR %s

# CHECK-ERR: error: invalid operand for instruction
ssload a0

# CHECK-ERR: error: invalid operand for instruction
sspopchk a1

# CHECK-ERR: error: invalid operand for instruction
c.sspush t0

# CHECK-ERR: error: invalid operand for instruction
c.sspopchk ra

# CHECK-ERR: error: immediate must be an integer in the range [1, 31]
sspinc 32

# CHECK-ERR: error: invalid operand for instruction
sspush a0

# CHECK-ERR: error: invalid operand for instruction
ssprr zero

# CHECK-ERR: error: invalid operand for instruction
ssamoswap zero, x0, (a0)
