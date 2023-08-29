// This test verifies SVE2p1 implies bf16.

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1 2>&1 < %s \
// RUN:        | FileCheck %s

bfcvt   z0.h, p0/m, z1.s
// CHECK: bfcvt   z0.h, p0/m, z1.s