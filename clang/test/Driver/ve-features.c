// RUN: %clang --target=ve-unknown-linux-gnu -### %s -mvevpu 2>&1 | FileCheck %s -check-prefix=VEVPU
// RUN: %clang --target=ve-unknown-linux-gnu -### %s -mno-vevpu 2>&1 | FileCheck %s -check-prefix=NO-VEVPU

// VEVPU: "-target-feature" "+vpu"
// NO-VEVPU-NOT: "-target-feature" "+vpu"
