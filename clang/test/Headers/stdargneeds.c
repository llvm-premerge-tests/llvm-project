// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds0.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds1.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds2.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds3.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds4.c
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.9.0 -verify -Werror=implicit-function-declaration -std=c89 %t/stdargneeds5.c

// Split the file so that the "implicitly declaring library function" errors get repeated.
// Use C89 to verify that __need_ can be used to get types that wouldn't normally be available.

//--- stdargneeds0.c
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared}}
    va_list v; // expected-error{{undeclared}}
    va_start(v, p); // expected-error{{implicit}} expected-note{{va_start}} expected-error{{undeclared}}
    int i = va_arg(v, int); // expected-error{{implicit}} expected-error{{expression}} expected-error{{undeclared}}
    va_end(v); // expected-error{{implicit}} expected-note{{va_end}} expected-error{{undeclared}}
    __va_copy(g, v); // expected-error{{implicit}} expected-error{{undeclared}} expected-error{{undeclared}}
    va_copy(g, v); // expected-error{{implicit}} expected-note{{va_copy}} expected-error{{undeclared}} expected-error{{undeclared}}
}

//--- stdargneeds1.c
#define __need___va_list
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v; // expected-error{{undeclared}}
    va_start(v, p); // expected-error{{implicit}} expected-note{{va_start}} expected-error{{undeclared}}
    int i = va_arg(v, int); // expected-error{{implicit}} expected-error{{expression}} expected-error{{undeclared}}
    va_end(v); // expected-error{{implicit}} expected-note{{va_end}} expected-error{{undeclared}}
    __va_copy(g, v); // expected-error{{implicit}} expected-error{{undeclared}}
    va_copy(g, v); // expected-error{{implicit}} expected-note{{va_copy}} expected-error{{undeclared}}
}

//--- stdargneeds2.c
#define __need_va_list
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared}}
    va_list v;
    va_start(v, p); // expected-error{{implicit}} expected-note{{va_start}}
    int i = va_arg(v, int); // expected-error{{implicit}} expected-error{{expression}}
    va_end(v); // expected-error{{implicit}} expected-note{{va_end}}
    __va_copy(g, v); // expected-error{{implicit}} expected-error{{undeclared}}
    va_copy(g, v); // expected-error{{implicit}} expected-note{{va_copy}} expected-error{{undeclared}}
}

//--- stdargneeds3.c
#define __need_va_list
#define __need_va_arg
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g; // expected-error{{undeclared}}
    va_list v;
    va_start(v, p);
    int i = va_arg(v, int);
    va_end(v);
    __va_copy(g, v); // expected-error{{implicit}} expected-error{{undeclared}}
    va_copy(g, v); // expected-error{{implicit}} expected-note{{va_copy}} expected-error{{undeclared}}
}

//--- stdargneeds4.c
#define __need___va_list
#define __need_va_list
#define __need___va_copy
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p); // expected-error{{implicit}} expected-note{{va_start}}
    int i = va_arg(v, int); // expected-error{{implicit}} expected-error{{expression}}
    va_end(v); // expected-error{{implicit}} expected-note{{va_end}}
    __va_copy(g, v);
    va_copy(g, v); // expected-error{{implicit}} expected-note{{va_copy}}
}

//--- stdargneeds5.c
#define __need___va_list
#define __need_va_list
#define __need_va_copy
#include <stdarg.h>
static void f(int p, ...) {
    __gnuc_va_list g;
    va_list v;
    va_start(v, p); // expected-error{{implicit}} expected-note{{va_start}}
    int i = va_arg(v, int); // expected-error{{implicit}} expected-error{{expression}}
    va_end(v); // expected-error{{implicit}} expected-note{{va_end}}
    __va_copy(g, v); // expected-error{{implicit}}
    va_copy(g, v);
}
