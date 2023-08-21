// By default, we don't pass any -fautoimport to -cc1, as that's the default.
//
// RUN: %clang --target=x86_64-w64-windows-gnu -### %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s
// RUN: %clang --target=x86_64-w64-windows-gnu -fno-autoimport -fautoimport -### %s 2>&1 | FileCheck --check-prefixes=DEFAULT %s
// DEFAULT: "-cc1"
// DEFAULT-NOT: no-autoimport
// DEFAULT-NOT: --disable-auto-import

// When compiling with -fno-autoimport, we pass -fno-autoimport to -cc1
// and --disable-auto-import to the linker.
//
// RUN: %clang --target=x86_64-w64-windows-gnu -fautoimport -fno-autoimport -### %s 2>&1 | FileCheck --check-prefixes=NO_AUTOIMPORT %s
// NO_AUTOIMPORT: "-cc1"
// NO_AUTOIMPORT: "-fno-autoimport"
// NO_AUTOIMPORT: "--disable-auto-import"
