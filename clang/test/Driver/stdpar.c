// RUN: %clang -### -stdpar --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=STDPAR-MISSING-LIB %s
// STDPAR-MISSING-LIB: error: cannot find HIP Standard Parallelism Acceleration library; provide it via '--stdpar-path'

// RUN: %clang -### --stdpar --stdpar-path=%S/Inputs/stdpar \
// RUN:   --stdpar-thrust-path=%S/Inputs/stdpar/thrust \
// RUN:   --stdpar-prim-path=%S/Inputs/stdpar/prim --compile %s 2>&1 | \
// RUN:   FileCheck --check-prefix=STDPAR-COMPILE %s
// STDPAR-COMPILE: "-x" "hip"
// STDPAR-COMPILE: "-idirafter" "{{.*/thrust}}"
// STDPAR-COMPILE: "-idirafter" "{{.*/prim}}"
// STDPAR-COMPILE: "-idirafter" "{{.*/Inputs/stdpar}}"
// STDPAR-COMPILE: "-include" "stdpar_lib.hpp"

// RUN: touch %t.o
// RUN: %clang -### -stdpar %t.o 2>&1 | FileCheck --check-prefix=STDPAR-LINK %s
// STDPAR-LINK: "-rpath"
// STDPAR-LINK: "-l{{.*hip.*}}"
