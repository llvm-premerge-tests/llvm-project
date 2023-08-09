// UNSUPPORTED: system-aix

// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

namespace NS1 { template <typename T> struct S {}; }
namespace NS2 { struct A { public: using S = int; }; }
namespace NS2 { A::S f(A::S a); }

// CHECK-NOT: error
