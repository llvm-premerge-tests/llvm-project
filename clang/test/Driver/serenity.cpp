// UNSUPPORTED: system-windows

/// Test a cross compiler.
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform --unwindlib=platform 2>&1 | FileCheck %s --check-prefix=SERENITY_x86_64
// SERENITY_x86_64:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// SERENITY_x86_64:      "-internal-isystem"
// SERENITY_x86_64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// SERENITY_x86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/include/c++/v1"
// SERENITY_x86_64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// SERENITY_x86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/local/include"
// SERENITY_x86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/include"
// SERENITY_x86_64:      "-L
// SERENITY_x86_64-SAME: {{^}}[[SYSROOT]]/usr/lib"

/// Loader name is the same for all architectures
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform --unwindlib=platform 2>&1 | FileCheck %s --check-prefix=DYNAMIC_LOADER
// RUN: %clang -### %s --target=aarch64-pc-serenity --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform --unwindlib=platform 2>&1 | FileCheck %s --check-prefix=DYNAMIC_LOADER
// RUN: %clang -### %s --target=riscv64-pc-serenity --sysroot= \
// RUN:   --stdlib=platform --rtlib=platform --unwindlib=platform 2>&1 | FileCheck %s --check-prefix=DYNAMIC_LOADER
// DYNAMIC_LOADER: "-dynamic-linker" "/usr/lib/Loader.so"

/// -r suppresses -dynamic-linker, default -l, and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform --rtlib=platform -r 2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE-NOT:  "-dynamic-linker"
// RELOCATABLE:      "-internal-isystem"
// RELOCATABLE-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// RELOCATABLE:      "-L
// RELOCATABLE-SAME: {{^}}[[SYSROOT]]/usr/lib"
// RELOCATABLE-NOT:  "-l
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
