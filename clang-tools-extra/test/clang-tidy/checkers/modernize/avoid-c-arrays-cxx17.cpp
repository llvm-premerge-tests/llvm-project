// RUN: %check_clang_tidy -std=c++17 %s modernize-avoid-c-arrays %t

//CHECK-FIXES: #include <array>

template <class T1, class T2>
struct Pair {
  T1 t1;
  T2 t2;
};

struct Obj1 {};
struct Obj2 {
  Obj2(const Obj1& = {});
};

struct StringRef {
  StringRef(const char*);
  StringRef(const StringRef&);
};

void ctad_replacements() {
  const int ci1{}, ci2{};
  int i1, i2;

  int ar[10];
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 10> ar;

  int init[] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init = {1,2,3};
  int init2[3] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> init2 = { {1,2,3} };
  char init3[] = "abcdef";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 7> init3 = { "abcdef" };
  char init4[] = "";
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 1> init4 = { "" };
  char init5[] = {"abc"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 4> init5 = { {"abc"} };
  char init6[] = {"abc" "def"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<char, 7> init6 = { {"abc" "def"} };
  int init7[] = {'c',2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<int, 3> init7 = { {'c',2,3} };
  const int init8[] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init8 = {1,2,3};
  int const init9[] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init9 = {1,2,3};
  int const init10[] = {i1, i2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init10 = {i1, i2};
  int const init11[] = {ci1, ci2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init11 = {ci1, ci2};
  int init12[] = {ci1, ci2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init12 = {ci1, ci2};
  constexpr volatile int static const init13[] = {1,2,3};
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: constexpr volatile static const std::array init13 = {1,2,3};
  const Obj1 init14[] = {Obj1(), Obj1()};
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init14 = {Obj1(), Obj1()};

  using IntPair = Pair<int,int>;
  IntPair init15[] = {IntPair{1,2}, IntPair{3,4}};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init15 = {IntPair{1,2}, IntPair{3,4}};
  using IntPair = Pair<int,int>;
  IntPair init16[] = { {1,2}, {3,4} };
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<IntPair, 2> init16 = { { {1,2}, {3,4} } };

  StringRef init17[] = {"a", "b"};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<StringRef, 2> init17 = { {"a", "b"} };
  StringRef sr1(""),sr2("");
  StringRef init18[] = {sr1, sr2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init18 = {sr1, sr2};
  StringRef init19[] = {StringRef(""), StringRef("")};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init19 = {StringRef(""), StringRef("")};
  StringRef init20[] = { {""} };
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array<StringRef, 1> init20 = { { {""} } };
  StringRef const init21[] = { StringRef{""} };
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: const std::array init21 = { StringRef{""} };

  int x0,x1;
  int* init22[] = {&x0, &x1};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init22 = {&x0, &x1};
  int*init23[] = {&x0, &x1};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: std::array init23 = {&x0, &x1};
  static thread_local int* const init24[] = {&x0, &x1};
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: static thread_local const std::array init24 = {&x0, &x1};
}

void replace_cv_within_type() {
  // FIXME: clang's AST TypeLoc etc do not give SourceLocations of individual
  // qualifiers etc. In combined types (e.g. 'unsigned int') with a cv
  // qualifier in the middle of the type (e.g., 'unsigned volatile int'), we
  // do not support pulling the qualifier out to appear before the std::array
  // decl.

  unsigned const int ui1{}, ui2{};

  unsigned volatile int ar1[] = {1u,2u};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: unsigned volatile int ar1[] = {1u,2u};
  volatile unsigned int ar2[] = {1u,2u};
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array ar2 = {1u,2u};
  unsigned int volatile ar3[] = {1u,2u};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array ar3 = {1u,2u};
  unsigned const int* volatile ar4[] = {&ui1, &ui2};
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not declare C-style arrays, use std::array<> instead
  // CHECK-FIXES: volatile std::array ar4 = {&ui1, &ui2};
}
