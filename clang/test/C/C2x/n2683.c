// RUN: %clang_cc1 -verify -ffreestanding -std=c23 %s

/* WG14 N2683: Clang 18
 * Define several macros for performing checked integer arithmetic
 */
#include <stdckdint.h>

void test_semantic() {
    _BitInt(33) a33 = 1;
    char char_var = 'd'; // The ascii value of `d` is 100
    bool bool_var = 1;
    enum week{Mon, Tue, Wed};
    enum week day = Mon;
    int result;

    bool flag_add = ckd_add(&result, a33, char_var);
    bool flag_sub = ckd_sub(&result, bool_var, day);
    bool flag_mul = ckd_mul(&result, day, char_var);
}

void test_invalid_input() {
    _BitInt(33) a33 = 1;
    char char_var = 'd'; // The ascii value of `d` is 100
    bool bool_var = 1;
    enum week{Mon, Tue, Wed};
    enum week day = Mon;
    char plain_char[] = {U'牛'}; /* expected-warning {{implicit conversion from 'unsigned int' to 'char' changes value from 29275 to 91}}  */
    bool flag_invalid = ckd_sub(&plain_char, char_var, day); /* expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('char (*)[1]' invalid)}} */
    bool flag_noptr = ckd_add(day, a33, bool_var); /* expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('enum week' invalid)}} */
}
