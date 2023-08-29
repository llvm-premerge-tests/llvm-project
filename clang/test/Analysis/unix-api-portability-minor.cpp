// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.APIPortabilityMinor -verify %s

#define NULL ((void*)0)

struct FILE_t;
typedef struct FILE_t FILE;

typedef __typeof(sizeof(int)) size_t;

int printf(const char *, ... );
int fprintf(FILE *, const char *, ...);
int sprintf(char *, const char *, ...);
int snprintf(char *, size_t, const char *, ...);

// Test basic case for the whole print family.
void test_printf_family_pointer_conversion_specifier_null(FILE *file, char *buf, size_t buf_size, char *format) {
  printf(format, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  fprintf(file, format, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  sprintf(buf, format, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  snprintf(buf, buf_size, format, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
}

// Test builtin null pointer type is handled correctly.
void test_printf_pointer_conversion_specifier_null_nullptr(char *format) {
  printf(format, nullptr); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
}

// Test various argument indexes.
void test_printf_pointer_conversion_specifier_null_various_arguments(char *format) {
  printf(format, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  printf(format, 1, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  printf(format, 1, NULL, 2); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  printf(format, NULL, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  printf(format, NULL, 1, NULL); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
  printf(format, 0); // no-warning
}

// Test pointer constraints.
void printf_pointer_conversion_specifier_null_pointer_constraints(char *format, int *pointer1, int *pointer2) {
  // Unknown pointer should not rase warning.
  printf(format, pointer1); // no-warning
  // Pointer argument should not get constrained after the check.
  *pointer1 = 777; // no-warning

  if (pointer2 != NULL) {
    printf(format, pointer1); // no-warning
    return;
  }
  printf(format, pointer2); // expected-warning{{The result of passing a null pointer to the pointer conversion specifier of the printf family of functions is implementation defined}}
}
