// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c99 -std=c99 %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c11 -std=c11 %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c23 -std=c23 %s

struct astruct { char member; };

ptrdiff_t p0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
size_t s0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
rsize_t r0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
wchar_t wc0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
void *v0 = NULL; // c99-error{{undeclared}} c11-error{{undeclared}} c23-error{{undeclared}}
nullptr_t n0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
static void f0(void) { unreachable(); } // c99-error{{undeclared}} c11-error{{undeclared}} c23-error{{undeclared}}
max_align_t m0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
size_t o0 = offsetof(struct astruct, member); // c99-error{{unknown}} c99-error{{undeclared}} c99-error{{expected expression}} c99-error{{undeclared}} \
                                                 c11-error{{unknown}} c11-error{{undeclared}} c11-error{{expected expression}} c11-error{{undeclared}} \
                                                 c23-error{{unknown}} c23-error{{undeclared}} c23-error{{expected expression}} c23-error{{undeclared}}
wint_t wi0; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}

// rsize_t will only be defined if __STDC_WANT_LIB_EXT1__ is set to >= 1.
// It would be nice to test the default undefined behavior, but that emits
// a note coming from stddef.h "rsize_t, did you mean size_t?" that can't be
// `c99-note`d away.
#define __STDC_WANT_LIB_EXT1__ 1
#include <stddef.h>

ptrdiff_t p1;
size_t s1;
rsize_t r1;
wchar_t wc1;
void *v1 = NULL;
nullptr_t n1; // c99-error{{unknown}} c11-error{{unknown}}
static void f1(void) { unreachable(); } // c99-error{{undeclared}} c11-error{{undeclared}}
max_align_t m1; // c99-error{{unknown}}
size_t o1 = offsetof(struct astruct, member);
wint_t wi1; // c99-error{{unknown}} c11-error{{unknown}} c23-error{{unknown}}
