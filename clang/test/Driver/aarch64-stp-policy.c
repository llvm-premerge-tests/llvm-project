// RUN: %clang -### -target aarch64 -aarch64-stp-policy=always %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALWAYS %s
// RUN: %clang -### -target aarch64 -aarch64-stp-policy=aligned %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALIGNED %s
// RUN: %clang -### -target aarch64 -aarch64-stp-policy=never %s -c 2>&1 | FileCheck -check-prefix=CHECK-NEVER %s
// RUN: %clang -### -target aarch64 -aarch64-stp-policy=default %s -c 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN: not %clang -### -target aarch64 -aarch64-stp-policy=def %s -c 2>&1 | FileCheck -check-prefix=CHECK-ARGUMENT %s
// RUN: not %clang -c -target x86-64 -aarch64-stp-policy=aligned %s 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s

// CHECK-ALWAYS: "-aarch64-stp-policy=always"
// CHECK-ALIGNED: "-aarch64-stp-policy=aligned"
// CHECK-NEVER: "-aarch64-stp-policy=never"
// CHECK-DEFAULT: "-aarch64-stp-policy=default"
// CHECK-ARGUMENT: clang: error: unsupported argument 'def' to option '-aarch64-stp-policy='
// CHECK-TRIPLE: clang: error: unsupported option '-aarch64-stp-policy=' for target
