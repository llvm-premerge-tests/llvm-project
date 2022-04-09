// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -mllvm -enable-boolean-counters=true -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name return.c %s | FileCheck %s

// CHECK: _Z6testIfi
int testIf(int x) { // CHECK-NEXT: File 0, [[@LINE]]:19 -> [[@LINE+10]]:2 = #0
                    // CHECK-NEXT: File 0, [[@LINE+5]]:7 -> [[@LINE+5]]:13 = #0
                    // CHECK-NEXT: Gap,File 0, [[@LINE+4]]:14 -> [[@LINE+5]]:5 = #2
                    // CHECK-NEXT: File 0, [[@LINE+4]]:5 -> [[@LINE+4]]:16 = #2
                    // CHECK-NEXT: File 0, [[@LINE+5]]:3 -> [[@LINE+5]]:16 = #1
  int result = 0;
  if (x == 0)
    result = -1;

  return result;
}

// CHECK-NEXT: _Z10testIfElsei
int testIfElse(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+13]]:2 = #0
                        // CHECK-NEXT: File 0, [[@LINE+7]]:7 -> [[@LINE+7]]:12 = #0
                        // CHECK-NEXT: Gap,File 0, [[@LINE+6]]:13 -> [[@LINE+7]]:5 = #1
                        // CHECK-NEXT: File 0, [[@LINE+6]]:5 -> [[@LINE+6]]:15 = #1
                        // CHECK-NEXT: Gap,File 0, [[@LINE+5]]:16 -> [[@LINE+7]]:5 = #2
                        // CHECK-NEXT: File 0, [[@LINE+6]]:5 -> [[@LINE+6]]:19 = #2
                        // CHECK-NEXT: File 0, [[@LINE+6]]:3 -> [[@LINE+6]]:16 = (#1 || #2)
  int result = 0;
  if (x < 0)
    result = 0;
  else
    result = x * x;
  return result;
}

// CHECK-NEXT: _Z16testIfElseReturni
int testIfElseReturn(int x) { // CHECK-NEXT: File 0, [[@LINE]]:29 -> [[@LINE+14]]:2 = #0
                              // CHECK-NEXT: File 0, [[@LINE+8]]:7 -> [[@LINE+8]]:12 = #0
                              // CHECK-NEXT: Gap,File 0, [[@LINE+7]]:13 -> [[@LINE+8]]:5 = #1
                              // CHECK-NEXT: File 0, [[@LINE+7]]:5 -> [[@LINE+7]]:19 = #1
                              // CHECK-NEXT: Gap,File 0, [[@LINE+6]]:20 -> [[@LINE+8]]:5 = #2
                              // CHECK-NEXT: File 0, [[@LINE+7]]:5 -> [[@LINE+7]]:13 = #2
                              // CHECK-NEXT: Gap,File 0, [[@LINE+6]]:14 -> [[@LINE+7]]:3 = #1
                              // CHECK-NEXT: File 0, [[@LINE+6]]:3 -> [[@LINE+6]]:16 = #1
  int result = 0;
  if (x > 0)
    result = x * x;
  else
    return 0;
  return result;
}

// CHECK-NEXT: _Z10testSwitchi
int testSwitch(int x) { // CHECK-NEXT: File 0, [[@LINE]]:23 -> [[@LINE+22]]:2 = #0
                        // CHECK-NEXT: Gap,File 0, [[@LINE+9]]:14 -> [[@LINE+17]]:15 = 0
                        // CHECK-NEXT: File 0, [[@LINE+9]]:3 -> [[@LINE+11]]:10 = #2
                        // CHECK-NEXT: Gap,File 0, [[@LINE+10]]:11 -> [[@LINE+11]]:3 = 0
                        // CHECK-NEXT: File 0, [[@LINE+10]]:3 -> [[@LINE+12]]:10 = #3
                        // CHECK-NEXT: Gap,File 0, [[@LINE+11]]:11 -> [[@LINE+12]]:3 = 0
                        // CHECK-NEXT: File 0, [[@LINE+11]]:3 -> [[@LINE+12]]:15 = #4
                        // CHECK-NEXT: Gap,File 0, [[@LINE+12]]:4 -> [[@LINE+14]]:3 = #1
                        // CHECK-NEXT: File 0, [[@LINE+13]]:3 -> [[@LINE+13]]:16 = #1
  int result;
  switch (x) {
  case 1:
    result = 1;
    break;
  case 2:
    result = 2;
    break;
  default:
    result = 0;
  }

  return result;
}

// CHECK-NEXT: _Z9testWhilei
int testWhile(int x) { // CHECK-NEXT: File 0, [[@LINE]]:22 -> [[@LINE+13]]:2 = #0
                       // CHECK-NEXT: File 0, [[@LINE+6]]:10 -> [[@LINE+6]]:16 = #0
                       // CHECK-NEXT: Gap,File 0, [[@LINE+5]]:17 -> [[@LINE+5]]:18 = #2
                       // CHECK-NEXT: File 0, [[@LINE+4]]:18 -> [[@LINE+7]]:4 = #2
                       // CHECK-NEXT: File 0, [[@LINE+8]]:3 -> [[@LINE+8]]:13 = #1
  int i = x;
  int sum = 0;
  while (i < 10) {
    sum += i;
    i++;
  }

  return sum;
}

// CHECK-NEXT: _Z12testContinuei
int testContinue(int x) { // CHECK-NEXT: File 0, [[@LINE]]:25 -> [[@LINE+21]]:2 = #0
                          // CHECK-NEXT: File 0, [[@LINE+12]]:10 -> [[@LINE+12]]:16 = ((#0 || #3) || #4)
                          // CHECK-NEXT: Gap,File 0, [[@LINE+11]]:17 -> [[@LINE+11]]:18 = #2
                          // CHECK-NEXT: File 0, [[@LINE+10]]:18 -> [[@LINE+15]]:4 = #2
                          // CHECK-NEXT: File 0, [[@LINE+10]]:9 -> [[@LINE+10]]:15 = #2
                          // CHECK-NEXT: Gap,File 0, [[@LINE+9]]:16 -> [[@LINE+10]]:7 = #4
                          // CHECK-NEXT: File 0, [[@LINE+9]]:7 -> [[@LINE+9]]:15 = #4
                          // CHECK-NEXT: Gap,File 0, [[@LINE+8]]:16 -> [[@LINE+9]]:5 = #3
                          // CHECK-NEXT: File 0, [[@LINE+8]]:5 -> [[@LINE+10]]:4 = #3
                          // CHECK-NEXT: Gap,File 0, [[@LINE+9]]:4 -> [[@LINE+11]]:3 = #1
                          // CHECK-NEXT: File 0, [[@LINE+10]]:3 -> [[@LINE+10]]:13 = #1
  int i = x;
  int sum = 0;
  while (i < 10) {
    if (i == 4)
      continue;
    sum += i;
    i++;
  }

  return sum;
}
