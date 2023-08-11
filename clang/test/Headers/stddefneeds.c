// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c99 -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c23 -std=c23 %s

// Use C99 to verify that __need_ can be used to get types that wouldn't normally be available.

struct astruct { char member; };

ptrdiff_t p0; // c99-error{{unknown}} c23-error{{unknown}}
size_t s0; // c99-error{{unknown}} c23-error{{unknown}}
rsize_t r0; // c99-error{{unknown}} c23-error{{unknown}}
wchar_t wc0; // c99-error{{unknown}} c23-error{{unknown}}
void *v0 = NULL; // c99-error{{undeclared}} c23-error{{undeclared}}
nullptr_t n0; // c99-error{{unknown}} c23-error{{unknown}}
static void f0(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m0; // c99-error{{unknown}} c23-error{{unknown}}
size_t o0 = offsetof(struct astruct, member); // c99-error{{unknown}} c99-error{{undeclared}} c99-error{{expected expression}} c99-error{{undeclared}} \
                                                 c23-error{{unknown}} c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi0; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_ptrdiff_t
#include <stddef.h>

ptrdiff_t p1;
size_t s1; // c99-error{{unknown}} c23-error{{unknown}}
rsize_t r1; // c99-error{{unknown}} c23-error{{unknown}}
wchar_t wc1; // c99-error{{unknown}} c23-error{{unknown}}
void *v1 = NULL; // c99-error{{undeclared}} c23-error{{undeclared}}
nullptr_t n1; // c99-error{{unknown}} c23-error{{unknown}}
static void f1(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m1; // c99-error{{unknown}} c23-error{{unknown}}
size_t o1 = offsetof(struct astruct, member); // c99-error{{unknown}} c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{unknown}} c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi1; // c99-error{{unknown}} c23-error{{unknown}}

// If you do these individually, the compiler will add a note
// coming from stddef.h "rsize_t, did you mean size_t?" that
// can't be `c99-note`d away.
#define __need_size_t
#define __need_rsize_t
#include <stddef.h>

ptrdiff_t p2;
size_t s2;
rsize_t r2;
wchar_t wc2; // c99-error{{unknown}} c23-error{{unknown}}
void *v2 = NULL; // c99-error{{undeclared}} c23-error{{undeclared}}
nullptr_t n2; // c99-error{{unknown}} c23-error{{unknown}}
static void f2(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m2; // c99-error{{unknown}} c23-error{{unknown}}
size_t o2 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi2; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_wchar_t
#include <stddef.h>

ptrdiff_t p3;
size_t s3;
rsize_t r3;
wchar_t wc3;
void *v3 = NULL; // c99-error{{undeclared}} c23-error{{undeclared}}
nullptr_t n3; // c99-error{{unknown}} c23-error{{unknown}}
static void f3(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m3; // c99-error{{unknown}} c23-error{{unknown}}
size_t o3 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi3; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_NULL
#include <stddef.h>

ptrdiff_t p4;
size_t s4;
rsize_t r4;
wchar_t wc4;
void *v4 = NULL;
nullptr_t n4; // c99-error{{unknown}} c23-error{{unknown}}
static void f4(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m4; // c99-error{{unknown}} c23-error{{unknown}}
size_t o4 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi4; // c99-error{{unknown}} c23-error{{unknown}}

#if __STDC_VERSION__ >= 202311L
// __need_nullptr_t generates an error in <C23 because its definition
// depends on nullptr, but the error gets attributed to the declaration
// in stddef.h so it can't be `c99-error`d.
#define __need_nullptr_t
#include <stddef.h>

ptrdiff_t p5;
size_t s5;
rsize_t r5;
wchar_t wc5;
void *v5 = NULL;
nullptr_t n5; // c99-error{{unknown}}
static void f5(void) { unreachable(); } // c99-error{{undeclared}} c23-error{{undeclared}}
max_align_t m5; // c99-error{{unknown}} c23-error{{unknown}}
size_t o5 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi5; // c99-error{{unknown}} c23-error{{unknown}}
#endif

#define __need_unreachable
#include <stddef.h>

ptrdiff_t p6;
size_t s6;
rsize_t r6;
wchar_t wc6;
void *v6 = NULL;
nullptr_t n6; // c99-error{{unknown}}
static void f6(void) { unreachable(); }
max_align_t m6; // c99-error{{unknown}} c23-error{{unknown}}
size_t o6 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi6; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_max_align_t
#include <stddef.h>

ptrdiff_t p7;
size_t s7;
rsize_t r7;
wchar_t wc7;
void *v7 = NULL;
nullptr_t n7; // c99-error{{unknown}}
static void f7(void) { unreachable(); }
max_align_t m7;
size_t o7 = offsetof(struct astruct, member); // c99-error{{undeclared}} c99-error{{expected expression}} \
                                                 c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi7; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_offsetof
#include <stddef.h>

ptrdiff_t p8;
size_t s8;
rsize_t r8;
nullptr_t n8; // c99-error{{unknown}}
static void f8(void) { unreachable(); }
wchar_t wc8;
void *v8 = NULL;
max_align_t m8;
size_t o8 = offsetof(struct astruct, member);
wint_t wi8; // c99-error{{unknown}} c23-error{{unknown}}

#define __need_wint_t
#include <stddef.h>

ptrdiff_t p9;
size_t s9;
rsize_t r9;
wchar_t wc9;
void *v9 = NULL;
nullptr_t n9; // c99-error{{unknown}}
static void f9(void) { unreachable(); }
max_align_t m9;
size_t o9 = offsetof(struct astruct, member);
wint_t wi9;
