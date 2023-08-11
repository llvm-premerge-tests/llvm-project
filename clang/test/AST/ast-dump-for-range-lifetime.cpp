// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s

namespace p2718r0 {
struct T {
  int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  T() {}
  ~T() {}
  const int *begin() const { return a; }
  const int *end() const { return a + 10; }
};

const T &f1(const T &t) { return t; }
T g() { return T(); }

void foo() {
  // CHECK: FunctionDecl {{.*}} foo 'void ()'
  // CHECK: `-CXXForRangeStmt {{.*}}
  // CHECK-NEXT:    |-<<<NULL>>>
  // CHECK-NEXT:    |-DeclStmt {{.*}}
  // CHECK-NEXT:    | `-VarDecl {{.*}} implicit used __range1 'const T &' cinit
  // CHECK-NEXT:    |   `-ExprWithCleanups {{.*}} 'const T':'const p2718r0::T' lvalue
  // CHECK-NEXT:    |     `-CallExpr {{.*}} 'const T':'const p2718r0::T' lvalue
  // CHECK-NEXT:    |       |-ImplicitCastExpr {{.*}} 'const T &(*)(const T &)' <FunctionToPointerDecay>
  // CHECK-NEXT:    |       | `-DeclRefExpr {{.*}} 'const T &(const T &)' lvalue Function {{.*}} 'f1' 'const T &(const T &)'
  // CHECK-NEXT:    |       `-MaterializeTemporaryExpr {{.*}} 'const T':'const p2718r0::T' lvalue extended by Var {{.*}} '__range1' 'const T &'
  // CHECK-NEXT:    |         `-ImplicitCastExpr {{.*}} 'const T':'const p2718r0::T' <NoOp>
  // CHECK-NEXT:    |           `-CXXBindTemporaryExpr {{.*}} <col:20, col:22> 'T':'p2718r0::T' (CXXTemporary {{.*}})
  // CHECK-NEXT:    |             `-CallExpr {{.*}} 'T':'p2718r0::T'
  // CHECK-NEXT:    |               `-ImplicitCastExpr {{.*}} 'T (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT:    |                 `-DeclRefExpr {{.*}} 'T ()' lvalue Function {{.*}} 'g' 'T ()'
  [[maybe_unused]] int sum = 0;
  for (auto e : f1(g()))
    sum += e;
}

struct LockGuard {
    LockGuard(int) {}

    ~LockGuard() {}
};

void f() {
  int v[] = {42, 17, 13};
  int M = 0;

  // CHECK: FunctionDecl {{.*}} f 'void ()'
  // CHECK: `-CXXForRangeStmt {{.*}}
  // CHECK-NEXT:   |-<<<NULL>>>
  // CHECK-NEXT:   |-DeclStmt {{.*}}
  // CHECK-NEXT:   | `-VarDecl {{.*}} col:16 implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:   |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:   |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:   |       |-CXXStaticCastExpr {{.*}}'void' static_cast<void> <ToVoid>
  // CHECK-NEXT:   |       | `-MaterializeTemporaryExpr {{.*}} 'LockGuard':'p2718r0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:   |       |   `-CXXFunctionalCastExpr {{.*}} 'LockGuard':'p2718r0::LockGuard' functional cast to LockGuard <ConstructorConversion>
  // CHECK-NEXT:   |       |     `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'p2718r0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:   |       |       `-CXXConstructExpr {{.*}} 'LockGuard':'p2718r0::LockGuard' 'void (int)'
  // CHECK-NEXT:   |       |         `-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT:   |       |           `-DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'M' 'int'
  // CHECK-NEXT:   |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for (int x : static_cast<void>(LockGuard(M)), v) // lock released in C++ 2020
  {
    LockGuard guard(M); // OK in C++ 2020, now deadlocks
  }
}

} // namespace p2718r0
