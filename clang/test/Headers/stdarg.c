// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c89 -Werror=implicit-function-declaration -std=c89 %t/stdarg0.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c99 -Werror=implicit-function-declaration -std=c99 %t/stdarg0.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c89 -Werror=implicit-function-declaration -std=c89 %t/stdarg1.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify=c99 -Werror=implicit-function-declaration -std=c99 %t/stdarg1.c

// Split the file so that the "implicitly declaring library function" errors get repeated.

//--- stdarg0.c
static void f(int p, ...) {
    __gnuc_va_list g; // c89-error{{undeclared}} c99-error{{undeclared}}
    va_list v; // c89-error{{undeclared}} c99-error{{undeclared}}
    va_start(v, p); // c89-error{{implicit}} c89-note{{va_start}} c89-error{{undeclared}} \
                       c99-error{{implicit}} c99-note{{va_start}} c99-error{{undeclared}}
    int i = va_arg(v, int); // c89-error{{implicit}} c89-error{{expression}} c89-error{{undeclared}} \
                               c99-error{{implicit}} c99-error{{expression}} c99-error{{undeclared}}
    va_end(v); // c89-error{{implicit}} c89-note{{va_end}} c89-error{{undeclared}} \
                  c99-error{{implicit}} c99-note{{va_end}} c99-error{{undeclared}}
    __va_copy(g, v); // c89-error{{implicit}} c89-error{{undeclared}} c89-error{{undeclared}} \
                        c99-error{{implicit}} c99-error{{undeclared}} c99-error{{undeclared}}
    va_copy(g, v); // c89-error{{implicit}} c89-note{{va_copy}} c89-error{{undeclared}} c89-error{{undeclared}} \
                      c99-error{{implicit}} c99-note{{va_copy}} c99-error{{undeclared}} c99-error{{undeclared}}
}

//--- stdarg1.c
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p);
    int i = va_arg(v, int);
    va_end(v);
    __va_copy(g, v);
    va_copy(g, v); // c89-error{{implicit}} c89-note{{va_copy}} c99-no-diagnostics
}
