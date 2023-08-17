// REQUIRES: x86-registered-target

// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT1 %s
// RUN: %clang -### -target x86_64 -fsplit-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-OPT2 %s
// RUN: %clang -### -target x86_64 -fprofile-use=default.profdata -fsplit-machine-functions -fno-split-machine-functions %s -c 2>&1 | FileCheck -check-prefix=CHECK-NOOPT %s

// CHECK-OPT1:       "-fsplit-machine-functions"
// CHECK-OPT2:       "-fsplit-machine-functions"
// CHECK-NOOPT-NOT:  "-fsplit-machine-functions"
