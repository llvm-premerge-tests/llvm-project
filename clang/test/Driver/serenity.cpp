// UNSUPPORTED: system-windows

/// Check default header and linker paths
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   2>&1 | FileCheck %s --check-prefix=PATHS_X86_64
// PATHS_X86_64:      "-resource-dir" "[[RESOURCE:[^"]+]]"
// PATHS_X86_64:      "-internal-isystem"
// PATHS_X86_64-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// PATHS_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/include/c++/v1"
// PATHS_X86_64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]/include"
// PATHS_X86_64-SAME: {{^}} "-internal-isystem" "[[SYSROOT:[^"]+]]/usr/include"
// PATHS_X86_64:      "-L
// PATHS_X86_64-SAME: {{^}}[[SYSROOT]]/usr/lib"

/// Check default linker args for each supported triple
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= 2>&1 | FileCheck %s --check-prefix=SERENITY_X86_64
// SERENITY_X86_64: "-cc1" "-triple" "x86_64-pc-serenity"
// SERENITY_X86_64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld"
// SERENITY_X86_64: "-pie"
// SERENITY_X86_64: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// SERENITY_X86_64: "-o" "a.out"
// SERENITY_X86_64: "-z" "pack-relative-relocs"
// SERENITY_X86_64: "crt0.o" "crti.o" "crtbeginS.o"
// SERENITY_X86_64: "-lc" "crtendS.o" "crtn.o"

// RUN: %clang -### %s --target=aarch64-pc-serenity --sysroot= 2>&1 | FileCheck %s --check-prefix=SERENITY_AARCH64
// SERENITY_AARCH64: "-cc1" "-triple" "aarch64-pc-serenity"
// SERENITY_AARCH64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld"
// SERENITY_AARCH64: "-pie"
// SERENITY_AARCH64: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// SERENITY_AARCH64: "-o" "a.out"
// SERENITY_AARCH64: "-z" "pack-relative-relocs"
// SERENITY_AARCH64: "crt0.o" "crti.o" "crtbeginS.o"
// SERENITY_AARCH64: "-lc" "crtendS.o" "crtn.o"

// RUN: %clang -### %s --target=riscv64-pc-serenity --sysroot= 2>&1 | FileCheck %s --check-prefix=SERENITY_RISCV64
// SERENITY_RISCV64: "-cc1" "-triple" "riscv64-pc-serenity"
// SERENITY_RISCV64: "{{(.*[^-.0-9A-Z_a-z])?}}ld.lld"
// SERENITY_RISCV64: "-pie"
// SERENITY_RISCV64: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// SERENITY_RISCV64: "-o" "a.out"
// SERENITY_RISCV64: "-z" "pack-relative-relocs"
// SERENITY_RISCV64: "crt0.o" "crti.o" "crtbeginS.o"
// SERENITY_RISCV64: "-lc" "crtendS.o" "crtn.o"

/// -static-pie suppresses -dynamic-linker
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
//      -static-pie 2>&1 | FileCheck %s --check-prefix=STATIC_PIE
// STATIC_PIE: "-pie" "-static"
// STATIC_PIE-NOT: "-dynamic-linker"
// STATIC_PIE: "--no-dynamic-linker" "-z" "text"
// STATIC_PIE:  "--eh-frame-hdr" "-z" "pack-relative-relocs"
// STATIC_PIE: "crt0.o" "crti.o" "crtbeginS.o"
// STATIC_PIE: "-lc" "crtendS.o" "crtn.o"

/// -shared forces use of shared crt files
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
//      -shared 2>&1 | FileCheck %s --check-prefix=SHARED
// SHARED: "-shared"
// SHARED:  "--eh-frame-hdr" "-z" "pack-relative-relocs"
// SHARED: "crt0_shared.o" "crti.o" "crtbeginS.o"
// SHARED: "-lc" "crtendS.o" "crtn.o"

/// -static forces use of static crt files
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
//      -static 2>&1 | FileCheck %s --check-prefix=STATIC
// STATIC: "-static"
// STATIC:  "--eh-frame-hdr" "-z" "pack-relative-relocs"
// STATIC: "crt0.o" "crti.o" "crtbegin.o"
// STATIC: "-lc" "crtend.o" "crtn.o"

/// -rdynamic passes -export-dynamic
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
//      -rdynamic 2>&1 | FileCheck %s --check-prefix=RDYNAMIC
// RDYNAMIC: "-pie" "-export-dynamic"
// RDYNAMIC: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// RDYNAMIC: "-o" "a.out"
// RDYNAMIC: "-z" "pack-relative-relocs"
// RDYNAMIC: "crt0.o" "crti.o" "crtbeginS.o"
// RDYNAMIC: "-lc" "crtendS.o" "crtn.o"

// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot= \
//      -no-pie -rdynamic 2>&1 | FileCheck %s --check-prefix=RDYNAMIC_NOPIE
// RDYNAMIC_NOPIE-NOT: "-pie"
// RDYNAMIC_NOPIE: "-export-dynamic"
// RDYNAMIC_NOPIE: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// RDYNAMIC_NOPIE: "-o" "a.out"
// RDYNAMIC_NOPIE: "-z" "pack-relative-relocs"
// RDYNAMIC_NOPIE: "crt0.o" "crti.o" "crtbeginS.o"
// RDYNAMIC_NOPIE: "-lc" "crtendS.o" "crtn.o"

/// -nostdlib suppresses default -l and crt*.o
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   -nostdlib --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=NOSTDLIB
// NOSTDLIB:      "-internal-isystem"
// NOSTDLIB-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// NOSTDLIB-NOT:  crt{{[^./]+}}.o
// NOSTDLIB:      "-L
// NOSTDLIB-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOSTDLIB-NOT:  "-l
// NOSTDLIB-NOT:  libclang_rt.builtins-x86_64.a
// NOSTDLIB-NOT:  crt{{[^./]+}}.o

// -nostartfiles suppresses crt*.o, but not default -l
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   -nostartfiles --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=NOSTARTFILES
// NOSTARTFILES:      "-internal-isystem"
// NOSTARTFILES-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// NOSTARTFILES-NOT:  crt{{[^./]+}}.o
// NOSTARTFILES:      "-L
// NOSTARTFILES-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOSTARTFILES:      "[[RESOURCE:[^"]+]]/lib/serenity/libclang_rt.builtins-x86_64.a"
// NOSTARTFILES:      "-lc"
// NOSTARTFILES-NOT:  crt{{[^./]+}}.o

/// -r suppresses -dynamic-linker, default -l, and crt*.o like -nostdlib.
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   -r --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=RELOCATABLE
// RELOCATABLE-NOT:  "-dynamic-linker"
// RELOCATABLE:      "-internal-isystem"
// RELOCATABLE-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
// RELOCATABLE:      "-L
// RELOCATABLE-SAME: {{^}}[[SYSROOT]]/usr/lib"
// RELOCATABLE-NOT:  "-l
// RELOCATABLE-NOT:  crt{{[^./]+}}.o
// RELOCATABLE-NOT:  libclang_rt.builtins-x86_64.a

/// -nolibc suppresses -lc but not other default -l
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   -nolibc --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=NOLIBC
// NOLIBC:      "-internal-isystem"
// NOLIBC-SAME: {{^}} "[[SYSROOT:[^"]+]]/usr/include/x86_64-pc-serenity/c++/v1"
// NOLIBC:      "[[SYSROOT:[^"]+]]/usr/lib/crt0.o" "[[SYSROOT:[^"]+]]/usr/lib/crti.o" "crtbeginS.o"
// NOLIBC:      "-L
// NOLIBC-SAME: {{^}}[[SYSROOT]]/usr/lib"
// NOLIBC-NOT:  "-lc"
// NOLIBC:      "[[RESOURCE:[^"]+]]/lib/serenity/libclang_rt.builtins-x86_64.a"
// NOLIBC:      "crtendS.o" "[[SYSROOT:[^"]+]]/usr/lib/crtn.o"

/// -fsanitize=undefined redirects to Serenity-custom UBSAN runtime
// RUN: %clang -### %s --target=x86_64-pc-serenity --sysroot=%S/Inputs/serenity_x86_64_tree \
// RUN:   -ccc-install-dir %S/Inputs/serenity_x86_64_tree/usr/local/bin -resource-dir=%S/Inputs/resource_dir \
// RUN:   -fsanitize=undefined --rtlib=compiler-rt 2>&1 | FileCheck %s --check-prefix=UBSAN
// UBSAN-NOT: "libclang_rt.ubsan{{[^./]+}}.a"
// UBSAN-NOT: "libclang_rt.ubsan{{[^./]+}}.so"
// UBSAN:     "-lubsan"

/// C++ stdlib behavior
// RUN: %clangxx -### %s --target=x86_64-pc-serenity --sysroot="" \
// RUN:   2>&1 | FileCheck %s --check-prefix=DEFAULT_LIBCXX
// DEFAULT_LIBCXX: "-dynamic-linker" "/usr/lib/Loader.so" "--eh-frame-hdr"
// DEFAULT_LIBCXX: "-z" "pack-relative-relocs"
// DEFAULT_LIBCXX: "crt0.o" "crti.o" "crtbeginS.o"
// DEFAULT_LIBCXX: "--push-state"
// DEFAULT_LIBCXX: "--as-needed"
// DEFAULT_LIBCXX: "-lc++"
// DEFAULT_LIBCXX: "--pop-state"
// DEFAULT_LIBCXX: "-lc" "crtendS.o" "crtn.o"

// RUN: %clangxx -### %s --target=x86_64-pc-serenity --sysroot="" \
// RUN:  -static 2>&1 | FileCheck %s --check-prefix=STATIC_LIBCXX
// STATIC_LIBCXX: "-z" "pack-relative-relocs"
// STATIC_LIBCXX: "crt0.o" "crti.o" "crtbegin.o"
// STATIC_LIBCXX: "--push-state"
// STATIC_LIBCXX: "--as-needed"
// STATIC_LIBCXX: "-lc++"
// STATIC_LIBCXX: "--pop-state"
// STATIC_LIBCXX: "-lc" "crtend.o" "crtn.o"

// RUN: %clangxx -### %s --target=x86_64-pc-serenity --sysroot="" \
// RUN:  -static-libstdc++ 2>&1 | FileCheck %s --check-prefix=STATIC_LIBSTDCXX
// STATIC_LIBSTDCXX: "-z" "pack-relative-relocs"
// STATIC_LIBSTDCXX: "crt0.o" "crti.o" "crtbeginS.o"
// STATIC_LIBSTDCXX: "--push-state"
// STATIC_LIBSTDCXX: "--as-needed"
// STATIC_LIBSTDCXX: "-Bstatic"
// STATIC_LIBSTDCXX: "-lc++"
// STATIC_LIBSTDCXX: "-Bdynamic"
// STATIC_LIBSTDCXX: "--pop-state"
// STATIC_LIBSTDCXX: "-lc" "crtendS.o" "crtn.o"

// RUN: %clangxx -### %s --target=x86_64-pc-serenity --sysroot="" \
// RUN:   -nostdlib++ 2>&1 | FileCheck %s --check-prefix=NO_LIBCXX
// NO_LIBCXX: "-z" "pack-relative-relocs"
// NO_LIBCXX: "crt0.o" "crti.o" "crtbeginS.o"
// NO_LIBCXX-NOT: "--push-state"
// NO_LIBCXX-NOT: "--as-needed"
// NO_LIBCXX-NOT: "-lc++"
// NO_LIBCXX-NOT: "--pop-state"
// NO_LIBCXX: "-lc" "crtendS.o" "crtn.o"
