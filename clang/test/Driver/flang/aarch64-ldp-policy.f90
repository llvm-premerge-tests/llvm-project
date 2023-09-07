! RUN: %clang -### --driver-mode=flang -target aarch64 -aarch64-ldp-policy=always %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALWAYS %s
! RUN: %clang -### --driver-mode=flang -target aarch64 -aarch64-ldp-policy=aligned %s -c 2>&1 | FileCheck -check-prefix=CHECK-ALIGNED %s
! RUN: %clang -### --driver-mode=flang -target aarch64 -aarch64-ldp-policy=never %s -c 2>&1 | FileCheck -check-prefix=CHECK-NEVER %s
! RUN: %clang -### --driver-mode=flang -target aarch64 -aarch64-ldp-policy=default %s -c 2>&1 | FileCheck -check-prefix=CHECK-DEFAULT %s
! RUN: not %clang -### --driver-mode=flang -target aarch64 -aarch64-ldp-policy=def %s -c 2>&1 | FileCheck -check-prefix=CHECK-ARGUMENT %s

! CHECK-ALWAYS: "-aarch64-ldp-policy=always"
! CHECK-ALIGNED: "-aarch64-ldp-policy=aligned"
! CHECK-NEVER: "-aarch64-ldp-policy=never"
! CHECK-DEFAULT: "-aarch64-ldp-policy=default"
! CHECK-ARGUMENT: clang: error: unsupported argument 'def' to option '-aarch64-ldp-policy='
