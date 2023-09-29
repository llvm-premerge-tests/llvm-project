// RUN: %clang_cc1 -verify=gnu -std=gnu++11 %s
// RUN: %clang_cc1 -verify=expected,cxx11 -Wvla -std=gnu++11 %s
// RUN: %clang_cc1 -verify=expected,cxx11 -std=c++11 %s
// RUN: %clang_cc1 -verify=expected,cxx98 -std=c++98 %s
// RUN: %clang_cc1 -verify=expected,off -std=c++11 -Wno-vla-extension-static-assert %s
// gnu-no-diagnostics

// Demonstrate that we do not diagnose use of VLAs by default in GNU mode, but
// we do diagnose them in C++ mode. Also note that we suggest use of
// static_assert, but only in C++11 and later and only if the warning group is
// not disabled.

// FIXME: it's not clear why C++98 mode does not emit the extra notes in the
// same way that C++11 mode does.
void func(int n) { // cxx11-note {{declared here}} off-note {{declared here}}
  int vla[n]; // expected-warning {{variable length arrays are a Clang extension}} \
                 cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                 off-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
}

void old_style_static_assert(int n) { // cxx11-note 3 {{declared here}} off-note {{declared here}}
  int array1[n != 12 ? 1 : -1]; // cxx11-warning {{variable length arrays are a Clang extension; did you mean to use 'static_assert'?}} \
                                   cxx98-warning {{variable length arrays are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array2[n != 12 ? -1 : 1]; // cxx11-warning {{variable length arrays are a Clang extension; did you mean to use 'static_assert'?}} \
                                   cxx98-warning {{variable length arrays are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  int array3[n != 12 ? 1 : n];  // expected-warning {{variable length arrays are a Clang extension}} \
                                   cxx11-note {{function parameter 'n' with unknown value cannot be used in a constant expression}} \
                                   off-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
}
