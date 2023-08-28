// RUN: mkdir -p %t.workdir && cd %t.workdir
// RUN: %clang -fsyntax-only %s -MJ - 2>&1 | FileCheck %s

// CHECK:      {
// CHECK-SAME: "directory": "{{[^"]*}}workdir",
// CHECK-SAME: "file": "{{[^"]*}}compilation_database_fsyntax_only.c"
// CHECK-NOT:  "output"
// CHECK-SAME: "arguments": [
// CHECK-NOT:    "-o"
// CHECK-SAME: ]
// CHECK-SAME: }


int main(void) {
  return 0;
}
