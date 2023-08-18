// REQUIRES: system-linux
// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: nvptx-registered-target

// clang does not accept "-o <output>" when generating multiple output
// files. Create temp directory and cd to it before invoking clang.
// RUN: rm -fr "%t" ; mkdir -p "%t" ; cd "%t"

// Check that -fsplit-machine-functions is passed to cuda and it
// causes a warning.
// RUN: %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions -S %s \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS1
// MFS1: warning: -fsplit-machine-functions is not valid for nvptx

// Check that -Xarch_host -fsplit-machine-functions does not cause any warning.
// RUN: %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN      --cuda-gpu-arch=sm_70 -x cuda -Xarch_host \
// RUN      -fsplit-machine-functions -S %s || { echo \
// RUN      "warning: -fsplit-machine-functions is not valid for" ; } \
// RUN      2>&1 | FileCheck %s --check-prefix=MFS2
// MFS2-NOT: warning: -fsplit-machine-functions is not valid for

// Check that -Xarch_device -fsplit-machine-functions does cause the warning.
// RUN: %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -Xarch_device \
// RUN:     -fsplit-machine-functions -S %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=MFS3
// MFS3: warning: -fsplit-machine-functions is not valid for

// Check that -fsplit-machine-functions -Xarch_device
// -fno-split-machine-functions has no warnings
// RUN: %clang --target=x86_64-unknown-linux-gnu -nogpulib -nogpuinc \
// RUN:     --cuda-gpu-arch=sm_70 -x cuda -fsplit-machine-functions \
// RUN:     -Xarch_device -fno-split-machine-functions -S %s \
// RUN:     || { echo "warning: -fsplit-machine-functions is not valid for"; } \
// RUN:     2>&1 | FileCheck %s --check-prefix=MFS4
// MFS4-NOT: warning: -fsplit-machine-functions is not valid for

// RUN: %clang -c --target=arm-unknown-linux-gnueabi -fsplit-machine-functions %s \
// RUN:     2>&1 | FileCheck -check-prefix=MFS5 %s
// MFS5: warning: -fsplit-machine-functions is not valid for arm
