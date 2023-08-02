// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-friend-template-other.cpp.ast %S/Inputs/ctu-friend-template-other.cpp
// RUN: cp %S/Inputs/ctu-friend-template.cpp.externalDefMap-dump.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -Werror=ctu \
// RUN:   -verify %s

// CHECK: CTU loaded AST file

#include "Inputs/ctu-friend-template.h"

void bar();

int main(){
	bar(); // expected-no-diagnostics
}
