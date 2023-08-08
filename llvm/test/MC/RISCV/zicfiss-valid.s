# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfiss,+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zicfiss,+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfiss,+c < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfiss,+c < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfiss -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: ssload x1
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssload x1

# CHECK-ASM-AND-OBJ: ssload x1
# CHECK-ASM: encoding: [0xf3,0x40,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssload ra

# CHECK-ASM-AND-OBJ: ssload x5
# CHECK-ASM: encoding: [0xf3,0x42,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssload x5

# CHECK-ASM-AND-OBJ: ssload x5
# CHECK-ASM: encoding: [0xf3,0x42,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssload t0

# CHECK-ASM-AND-OBJ: sspopchk x1
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk x1

# CHECK-ASM-AND-OBJ: sspopchk x1
# CHECK-ASM: encoding: [0x73,0xc0,0xc0,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk ra

# CHECK-ASM-AND-OBJ: sspopchk x5
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk x5

# CHECK-ASM-AND-OBJ: sspopchk x5
# CHECK-ASM: encoding: [0x73,0xc0,0xc2,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspopchk t0

# CHECK-ASM-AND-OBJ: sspinc 4
# CHECK-ASM: encoding: [0x73,0x40,0xd2,0x81]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspinc 4

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0x8a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush x1

# CHECK-ASM-AND-OBJ: sspush ra
# CHECK-ASM: encoding: [0x73,0x40,0x10,0x8a]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush ra

# check-asm-and-obj: sspush t0
# check-asm: encoding: [0x73,0x40,0x50,0x8a]
# check-no-ext: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush x5

# check-asm-and-obj: sspush t0
# check-asm: encoding: [0x73,0x40,0x50,0x8a]
# check-no-ext: error: instruction requires the following: 'Zicfiss' (Shadow stack)
sspush t0

# CHECK-ASM-AND-OBJ: ssprr ra
# CHECK-ASM: encoding: [0xf3,0x40,0x00,0x86]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssprr ra

# CHECK-ASM-AND-OBJ: ssamoswap t0, zero, (a0)
# CHECK-ASM: encoding: [0xf3,0x42,0x05,0x82]
# CHECK-NO-EXT: error: instruction requires the following: 'Zicfiss' (Shadow stack)
ssamoswap t0, x0, (a0)

# CHECK-ASM-AND-OBJ: c.sspush x1
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'C' (Compressed Instructions), 'Zicfiss' (Shadow stack)
c.sspush x1

# CHECK-ASM-AND-OBJ: c.sspush x1
# CHECK-ASM: encoding: [0x81,0x60]
# CHECK-NO-EXT: error: instruction requires the following: 'C' (Compressed Instructions), 'Zicfiss' (Shadow stack)
c.sspush ra

# CHECK-ASM-AND-OBJ: c.sspopchk x5
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'C' (Compressed Instructions), 'Zicfiss' (Shadow stack)
c.sspopchk x5

# CHECK-ASM-AND-OBJ: c.sspopchk x5
# CHECK-ASM: encoding: [0x81,0x62]
# CHECK-NO-EXT: error: instruction requires the following: 'C' (Compressed Instructions), 'Zicfiss' (Shadow stack)
c.sspopchk t0
