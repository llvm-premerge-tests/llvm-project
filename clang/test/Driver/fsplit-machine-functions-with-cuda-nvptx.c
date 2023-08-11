// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: shell

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions -S %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MFS2

// Check that -fsplit-machine-functions is passed to both x86 and cuda compilation and does not cause driver error.
// MFS2: -fsplit-machine-functions
// MFS2: -fsplit-machine-functions

// RUN:   %clang -### --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc --cuda-gpu-arch=sm_70 -x cuda -Xarch_host -fsplit-machine-functions -S %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MFS1

// Check that -Xarch_host -fsplit-machine-functions is passed only to native compilation.
// MFS1: "-target-cpu" "x86-64"{{.*}}"-fsplit-machine-functions"
// MFS1-NOT: "-target-cpu" "sm_70"{{.*}}"-fsplit-machine-functions"



