// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-output=text -verify \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-output=text -verify \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++14 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-output=text -verify \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++20 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow

// This test file verifies that the BitwiseShift checker does not crash or
// report false positives (at least on the cases that are listed here...)

// TEST OBVIOUSLY CORRECT CODE
//===----------------------------------------------------------------------===//

unsigned shift_unsigned(void) {
  // Shifts of unsigned LHS may overflow, even if the RHS is signed.
  // In shifts the type of the right operand does not affect the type of the
  // calculation and the result.
  return 1024u << 25ll; // no-warning
}

int shift_zeroes(void) {
  return 0 << 0; // no-warning
}

int no_info(int left, int right) {
  return left << right; // no-warning
}

int all_okay(int left, int right) {
  if (left < 0 || right < 0)
    return 42;
  return (left << right) + (left >> right); // no-warning
}

// TEST UNUSUAL CODE THAT SHOULD NOT CRASH
//===----------------------------------------------------------------------===//

__int128 large_left(void) {
  // Ensure that we do not crash when the left operand doesn't fit in 64 bits.
  return (__int128) 1 << 63 << 10 << 10; // no-crash
}

int large_right(void) {
  // Ensure that we do not crash when the right operand doesn't fit in 64 bits.
  return 1 << ((__int128) 1 << 118); // no-crash
  // expected-warning@-1 {{Left shift by '332306998946228968225951765070086144' overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand '332306998946228968225951765070086144' is not smaller than 32, the capacity of 'int'}}
}


void doubles_cast_to_integer(int *c) {
  *c = 1 << (int)1.5;          // no-crash
  *c = ((int)1.5) << 1;        // no-crash
  *c = ((int)1.5) << (int)1.5; // no-crash
}
