// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config eagerly-assume=false \
// RUN:    -verify=expected,c \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config eagerly-assume=false \
// RUN:    -verify=expected,cxx \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++14 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow

// Tests for validating the state updates provided by the BitwiseShift checker.
// These clang_analyzer_eval() tests are in a separate file because
// debug.ExprInspection repeats each 'warning' with an superfluous 'note', so
// note level output (-analyzer-output=text) is not enabled in this file.

void clang_analyzer_eval(int);

int state_update_generic(int left, int right) {
  int x = left << right;
  clang_analyzer_eval(left >= 0); // expected-warning {{TRUE}}
  clang_analyzer_eval(left > 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(right >= 0); // expected-warning {{TRUE}}
  clang_analyzer_eval(right > 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(right < 31); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(right < 32); // expected-warning {{TRUE}}
  // No similar upper bound on 'left':
  clang_analyzer_eval(left < 32); // expected-warning {{UNKNOWN}}
  // In fact, there is no upper bound at all:
  clang_analyzer_eval(left < 123456); // expected-warning {{UNKNOWN}}
  return x;
}

int state_update_exact_shift(int arg) {
  int x = 65535 << arg;
  clang_analyzer_eval(arg >= 0); // expected-warning {{TRUE}}
  clang_analyzer_eval(arg > 0); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(arg < 15); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(arg < 16); // c-warning {{TRUE}} cxx-warning {{UNKNOWN}}
  clang_analyzer_eval(arg <= 16); // expected-warning {{TRUE}}
  return x;
}
