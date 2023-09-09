// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: "", \
// RUN:  }}' --

// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: "none", \
// RUN:  }}' --

// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t \
// RUN:   -config='{CheckOptions: { \
// RUN:     bugprone-easily-swappable-parameters.MinimumLength: "null", \
// RUN:  }}' --

void a(int b, int c) {}
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 2 adjacent parameters of 'a' of similar type ('int') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:12: note: the first parameter in the range is 'b'
// CHECK-MESSAGES: :[[@LINE-3]]:19: note: the last parameter in the range is 'c'
