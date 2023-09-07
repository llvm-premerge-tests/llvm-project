// RUN: %clang -### -target aarch64 -aarch64-ldp-policy=always %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALWAYS %s
// RUN: %clang -### -target aarch64 -aarch64-ldp-policy=aligned %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALIGNED %s
// RUN: %clang -### -target aarch64 -aarch64-ldp-policy=never %s -c 2>&1 | FileCheck -check-prefix=CHECK-NEVER %s
// RUN: %clang -### -target aarch64 -aarch64-ldp-policy=default %s -c 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
// RUN: not %clang -### -target aarch64 -aarch64-ldp-policy=def %s -c 2>&1 | FileCheck -check-prefix=CHECK-ARGUMENT %s
// RUN: not %clang -c -target x86-64 -aarch64-ldp-policy=aligned %s 2>&1 | FileCheck -check-prefix=CHECK-TRIPLE %s

// CHECK-ALWAYS: "-aarch64-ldp-policy=always"
// CHECK-ALIGNED: "-aarch64-ldp-policy=aligned"
// CHECK-NEVER: "-aarch64-ldp-policy=never"
// CHECK-DEFAULT: "-aarch64-ldp-policy=default"
// CHECK-ARGUMENT: clang: error: unsupported argument 'def' to option '-aarch64-ldp-policy='
// CHECK-TRIPLE: clang: error: unsupported option '-aarch64-ldp-policy=' for target
