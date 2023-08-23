// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s

extern "C" {
void foo(int *ptr) {
  ptr[5] = 10;  // expected-warning{{unsafe buffer access through raw pointer parameter variable 'ptr'}}
}

void bar(int *ptr);

struct c_struct {
  char *name;
};
}

void bar(int *ptr) {
  ptr[5] = 10;  // expected-warning{{unsafe buffer access through raw pointer parameter variable 'ptr'}}
}

void call_foo(int *p) {
  foo(p);
  struct c_struct str;
  str.name[7] = 9;  // expected-warning{{unsafe buffer access through raw pointer member variable 'name'}}
  bar(p);
}
