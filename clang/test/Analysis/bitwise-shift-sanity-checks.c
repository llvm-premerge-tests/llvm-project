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

// expected-no-diagnostics

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

__int128 large_int(void) {
  // Ensure that we do not crash on values that do not fit in 64 bits.
  return (__int128) 1 << 63 << 10 << 10; // no-crash
}

void doubles_cast_to_integer(int *c) {
  *c = 1 << (int)1.5;          // no-crash
  *c = ((int)1.5) << 1;        // no-crash
  *c = ((int)1.5) << (int)1.5; // no-crash
}
