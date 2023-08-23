// RUN: %clang_cc1 -verify -ffreestanding -std=c23 %s

/* WG14 N2683: Clang 18
 * Towards Integer Safety
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

  // FIXME: add static_assert calls to check the resulting values for correctness
  // once the constant expression interpreter is able to handle the checked arithmetic
  // builtins in C. Currently, they're only a valid constant expression in C++ due to
  // looking for an ICE in C. Also all values in the tests of n2683_2.c should be checked.
}

void test_invalid_input() {
  _BitInt(33) a33 = 1;
  char char_var = 'd'; // The ascii value of `d` is 100
  bool bool_var = 1;
  const int const_result = 0;
  enum week{Mon, Tue, Wed};
  enum week day = Mon;
  short short_result = 0;
  char plain_char[] = {U'牛'}; /* expected-warning {{implicit conversion from 'unsigned int' to 'char' changes value from 29275 to 91}}  */
  bool flag_invalid = ckd_sub(&plain_char, char_var, day); /* expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('char (*)[1]' invalid)}} */
  bool flag_noptr = ckd_add(day, a33, bool_var); /* expected-error {{result argument to overflow builtin must be a pointer to a non-const integer ('enum week' invalid)}} */
  bool flag_noconst = ckd_add(&const_result, a33, plain_char); /* expected-error {{operand argument to overflow builtin must be an integer ('char *' invalid)}} */
  bool flag_noshort = ckd_mul(&short_result, bool_var, char_var); /* expected-warning {{'short' may not be suitable to hold the result of operating two 'int's}} */
}
