// RUN: %clang -### %s --target=x86_64-unknown-uefi \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// RUN: %clang -### %s --target=x86_64-uefi \
// RUN:     --sysroot=%S/platform -fuse-ld=lld 2>&1 \
// RUN:     | FileCheck -check-prefixes=CHECK %s
// CHECK: "-cc1"
// CHECK-SAME: "-triple" "x86_64-unknown-uefi"
// CHECK-SAME: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-SAME: "-mframe-pointer=all"
