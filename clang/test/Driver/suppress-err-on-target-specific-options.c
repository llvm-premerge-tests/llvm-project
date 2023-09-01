// REQUIRES: x86-registered-target
// REQUIRES: aarch64-registered-target

// RUN: %clang --target=aarch64-none-gnu --verbose -mcpu= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=aarch64-none-gnu --verbose -march= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=aarch64-none-gnu -### -mcpu= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=aarch64-none-gnu -### -march= 2>&1 | FileCheck %s --check-prefix=WARNING

// RUN: %clang --target=x86_64-unknown-linux-gnu --verbose -mcpu= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-unknown-linux-gnu --verbose -march= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-unknown-linux-gnu -### -mcpu= 2>&1 | FileCheck %s --check-prefix=WARNING
// RUN: %clang --target=x86_64-unknown-linux-gnu -### -march= 2>&1 | FileCheck %s --check-prefix=WARNING

// RUN: touch %t.c
// RUN: not %clang --target=x86_64-unknown-linux-gnu --verbose -mcpu=native %t.c -S 2>&1 | FileCheck %s --check-prefix=ERROR-X86
// RUN: not %clang --target=x86_64-unknown-linux-gnu -### -mcpu=native %t.c -S 2>&1 | FileCheck %s --check-prefix=ERROR-X86

// RUN: %clang --target=aarch64-none-gnu --verbose -mcpu=native %t.c -S 2>&1 | FileCheck %s --check-prefix=ERROR-AARCH64
// RUN: %clang --target=aarch64-none-gnu -### -mcpu=native %t.c 2>&1 -S | FileCheck %s --check-prefix=ERROR-AARCH64

// In situation when there is no compilation/linking clang should not emit error
// about target specific options, but just warn that are not used.
WARNING: warning: argument unused during compilation

ERROR-X86: error: unsupported option {{.*}} for target 'x86_64-unknown-linux-gnu'
ERROR-AARCH64-NOT: error: unsupported option {{.*}} for target 'aarch64-none-gnu'
