// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fbuiltin-headers-in-system-modules -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -verify=no-stddef-module -fno-modules-error-recovery
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/StdDef %s -fno-modules-error-recovery

#include "ptrdiff_t.h"

ptrdiff_t pdt;

// When builtin headers join system modules, stddef.h and its sub-headers have no
// header guards, and so are seen last by include_again.h, which takes all of their
// declarations including size_t even though size_t.h previously declared it.
// When builtin headers don't join the system modules and instead get their own
// modules, none of the stddef.h declarations go in the StdDef test module. size_t
// is then declared in both StdDef.SizeT and _Builtin_stddef.size_t. For the
// purposes of this test it doesn't matter which one gets reported, just as long
// as it isn't other.h or include_again.h.
size_t st; // no-stddef-module-error {{missing '#include "include_again.h"'; 'size_t' must be declared before it is used}} \
              yes-stddef-module-error {{missing '#include "size_t.h"'; 'size_t' must be declared before it is used}}
// no-stddef-module-note@__stddef_size_t.h:* {{here}}
// yes-stddef-module-note@size_t.h:* {{here}}

#include "include_again.h"

size_t st2;
