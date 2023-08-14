// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ast-dump -frecovery-ast %s | FileCheck %s

void foo();
void fun(int arg = foo());
//      CHECK: -ParmVarDecl {{.*}} <col:10, col:24> col:14 invalid arg 'int' cinit
// CHECK-NEXT:   -RecoveryExpr {{.*}} <col:18, col:24> 'int' contains-errors
