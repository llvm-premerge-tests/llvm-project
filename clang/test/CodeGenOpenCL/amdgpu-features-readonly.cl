// RUN: %clang_cc1 -triple amdgcn -target-feature +gds -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=GDS %s
// RUN: %clang_cc1 -triple amdgcn -target-feature +gws -o /dev/null %s 2>&1 \
// RUN:   | FileCheck --check-prefix=GWS %s

// GDS: warning: feature flag '+gds' is ignored since the feature is read only [-Winvalid-command-line-argument]
// GWS: warning: feature flag '+gws' is ignored since the feature is read only [-Winvalid-command-line-argument]

kernel void test() {}
