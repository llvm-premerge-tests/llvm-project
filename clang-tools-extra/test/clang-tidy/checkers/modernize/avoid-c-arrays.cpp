// RUN: %check_clang_tidy -std=c++11 %s modernize-avoid-c-arrays %t

//CHECK-FIXES: #include <array>

int a[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

int b[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

void foo() {
  int c[b[0]];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use std::vector<> instead

  using d = decltype(c);
  d e;
  // Semi-FIXME: we do not diagnose these last two lines separately,
  // because we point at typeLoc.getBeginLoc(), which is the decl before that
  // (int c[b[0]];), which is already diagnosed.
}

template <typename T, int Size>
class array {
  T d[Size];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};

array<int[4], 2> d;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: do not declare C-style arrays, use std::array<> instead

using k = int[4];
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: do not declare C-style arrays, use std::array<> instead

array<k, 2> dk;

template <typename T>
class unique_ptr {
  T *d;

  int e[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};

unique_ptr<int[]> d2;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead

using k2 = int[];
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead

unique_ptr<k2> dk2;

// Some header
extern "C" {

int f[] = {1, 2};

int j[1];

inline void bar() {
  {
    int j[j[0]];
  }
}

extern "C++" {
int f3[] = {1, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

int j3[1];
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: do not declare C-style arrays, use std::array<> instead

struct Foo {
  int f3[3] = {1, 2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  int j3[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};
}

struct Bar {

  int f[3] = {1, 2};

  int j[1];
};
}

template <class T> struct TStruct {};

void replacements() {
  int ar[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 10> ar;
  TStruct<int> ar2[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<TStruct<int>, 10> ar2;
  TStruct< int > ar3[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<TStruct< int >, 10> ar3;
  int * ar4[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int *, 10> ar4;
  int * /*comment*/ar5[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int *, 10> /*comment*/ar5;
  volatile const int * ar6[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<volatile const int *, 10> ar6;
  volatile int ar7[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array<int, 10> ar7;
  int const * ar8[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int const *, 10> ar8;
  int ar9[1];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 1> ar9;
  static int volatile constexpr ar10[10] = {};
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: static volatile constexpr std::array<int, 10> ar10 = {{[{][{]}}{{[}][}]}};
  thread_local int ar11[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: thread_local std::array<int, 10> ar11;
  thread_local/*a*/int/*b*/ar12[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: thread_local/*a*/std::array<int, 10>/*b*/ar12;
  /*a*/ int/*b*/ /*c*/*/*d*/ /*e*/ /*f*/ar13[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: /*a*/ std::array<int/*b*/ /*c*/*, 10>/*d*/ /*e*/ /*f*/ar13;
  TStruct<int*> ar14[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<TStruct<int*>, 10> ar14;
  volatile TStruct<const int*> ar15[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array<TStruct<const int*>, 10> ar15;
  TStruct<int const*> ar16[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<TStruct<int const*>, 10> ar16;
  TStruct<unsigned const int> ar17[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<TStruct<unsigned const int>, 10> ar17;
  volatile int static thread_local * ar18[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: static thread_local std::array<volatile int *, 10> ar18;

  // Note, there is a tab '\t' before the semicolon in the declaration below.
  int ar19[3]	;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> ar19	;

  int
  ar20[3];
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3>
  // CHECK-FIXES-NEXT: ar20;

  int init[] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> init = {{[{][{]}}1,2,3{{[}][}]}};
  int init2[3] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> init2 = {{[{][{]}}1,2,3{{[}][}]}};
  int init3[3] = {1,2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> init3 = {{[{][{]}}1,2{{[}][}]}};
  char init4[] = "abcdef";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 7> init4 = {"abcdef"};
  wchar_t init5[] = L"abcdef";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<wchar_t, 7> init5 = {L"abcdef"};
  char init6[] = "";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 1> init6 = {""};
  char init7[] = "\0";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 2> init7 = {"\0"};
  char init8[] = {"abc"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 4> init8 = {{[{][{]}}"abc"{{[}][}]}};
  char init9[] = R"LONG(a
  really
  long
  string)LONG";
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 27> init9 = {R"LONG(a
  // CHECK-FIXES-NEXT:   really
  // CHECK-FIXES-NEXT:   long
  // CHECK-FIXES-NEXT:   string)LONG"};
  char init10[] = {"abc" "def"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 7> init10 = {{[{][{]}}"abc" "def"{{[}][}]}};
  const char* init11 = {nullptr};
  const char* const init12[] = {"abc","def"};
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array<const char*, 2> init12 = {{[{][{]}}"abc","def"{{[}][}]}};
  const char*const init13[] = {"abc","def"};
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array<const char*, 2>init13 = {{[{][{]}}"abc","def"{{[}][}]}};

  // CHECK-MESSAGES: :[[@LINE+2]]:9: warning: do not declare C-style arrays, use std::array<> instead
#define NAME "ABC"
  const char init14[] = NAME
#if 1
    " "
#endif
    ;

  // Note: there are two tab '\t' characters between the 'int', '*', and 'init15' tokens below.
  int	*	init15[1] = {nullptr};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int	*, 1>	init15 = {{[{][{]}}nullptr{{[}][}]}};

  int two_d[10][5];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: int two_d[10][5];
  int three_d[10][5][3];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: int three_d[10][5][3];
}

void replace_cv_within_type() {
  // FIXME: clang's AST TypeLoc etc do not give SourceLocations of individual
  // qualifiers etc. In combined types (e.g. 'unsigned int') with a cv
  // qualifier in the middle of the type (e.g., 'unsigned volatile int'), we
  // do not support pulling the qualifier out to appear before the std::array
  // decl.

  unsigned volatile int ar1[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: unsigned volatile int ar1[10];
  volatile unsigned int ar2[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array<unsigned int, 10> ar2;
  unsigned int volatile ar3[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array<unsigned int, 10> ar3;
  unsigned const int* volatile ar4[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array<unsigned const int*, 10> ar4;
}

void consumes_ptr(int*);
void consumes_ptr(char*);
void replacement_with_refs() {
  int ar[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 10> ar;

  for (int i = 0; i != sizeof(ar)/sizeof(ar[0]); ++i) { ar[i]++; }
  // CHECK-FIXES: for (int i = 0; i != sizeof(ar)/sizeof(ar[0]); ++i) { ar[i]++; }
  for (int i: ar) {}
  // CHECK-FIXES: for (int i: ar) {}

  int* ptr = ar;
  // CHECK-FIXES: int* ptr = ar.begin();
  const int* cptr = ar;
  // CHECK-FIXES: const int* cptr = ar.begin();
  ptr = (ar+1);
  // CHECK-FIXES: ptr = (ar.begin()+1);

  int a;
  a = ar[0] + (ar)[1] + ((ar))[2];
  // CHECK-FIXES: a = ar[0] + (ar)[1] + ((ar))[2];
  a = 0[ar] + (0)[ar] + 0[(ar)];
  // CHECK-FIXES: a = ar[0] + ar[(0)] + (ar)[0];
  a = *(ar+2);
  // CHECK-FIXES: a = *(ar.begin()+2);
  consumes_ptr(ar);
  // CHECK-FIXES: consumes_ptr(ar.begin());

  unsigned sz = sizeof(ar) + sizeof((ar)) + sizeof(ar[0]) + sizeof ar;
  // CHECK-FIXES: unsigned sz = sizeof(ar) + sizeof((ar)) + sizeof(ar[0]) + sizeof ar;
}

void replacement_with_refs_string_array() {
  char str[] = {"abcdef"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 7> str = {{[{][{]}}"abcdef"{{[}][}]}};

  consumes_ptr(str);
  // CHECK-FIXES: consumes_ptr(str.begin());

  char* cp;
  cp = str + 0;
  // CHECK-FIXES: cp = str.begin() + 0;
  cp = str + 1;
  // CHECK-FIXES: cp = str.begin() + 1;
  cp = &str[0];
  // CHECK-FIXES: cp = &str[0];
}

void cannot_replace_reference_taken() {
  int ar[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  int (&ref)[10] = ar;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
}

struct SomeClass {
  void fn1(int);
  void fn2(int);
};
void fn1(int);
void fn2(int);
void cannot_replace_array_of_function_pointers() {
  void (*fns1[])(int) = {fn1, fn2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  using FnPtr = void(*)(int);
  FnPtr fns2[] = {fn1, fn2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  void (SomeClass::*fns3[])(int) = {&SomeClass::fn1, &SomeClass::fn2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead

  using MemFnPtr = void(SomeClass::*)(int);
  MemFnPtr fns4[] = {&SomeClass::fn1, &SomeClass::fn2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
}

void consume_array_ref(int (&)[10]);
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not declare C-style arrays, use std::array<> instead
void cannot_replace_reference_taken_in_call() {
  int ar[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  consume_array_ref(ar);
}

void vla_not_replaced(int n) {
  int ar[n];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C VLA arrays, use std::vector<> instead
}

struct Obj1 {};
struct Obj2 {
  Obj2(const Obj1& = {});
};
void init_expr_with_temp() {
  Obj2 ar[] = { {} };
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<Obj2, 1> ar = {{[{][{]}} {} {{[}][}]}};
}

struct AStruct {
  #define DEFINES_METHOD_WITH_ARRAY void method() { int ar[10]; }
  DEFINES_METHOD_WITH_ARRAY
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
};

void macros_not_replaced() {
  int d[3];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
#define EXPANDS_X(x) x
  EXPANDS_X(consumes_ptr(d));
}

void attrs_not_replaced() {
  [[maybe_unused]] int ar1[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: do not declare C-style arrays, use std::array<> instead
  __attribute__((__unused__)) int ar2[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: do not declare C-style arrays, use std::array<> instead
}
