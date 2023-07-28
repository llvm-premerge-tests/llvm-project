// PowerPC supports VLAs.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o %t-ppc-device.ll

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o %t-ppc-device.ll

// Nvidia GPUs don't support VLAs.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host-nvptx.bc
// RUN: %clang_cc1 -verify=unsupported,expected -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-nvptx.bc -o %t-nvptx-device.ll

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host-nvptx.bc

#pragma omp declare target
void declare(int arg) { // expected-note {{declared here}}
  int a[2];
  int vla[arg]; // unsupported-error {{variable length arrays are not supported for the current target}} \
                   expected-warning {{variable length arrays are a Clang extension}} \
                   expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}
}

void declare_parallel_reduction(int arg) {
  int a[2];

#pragma omp parallel reduction(+: a)
  { }

#pragma omp parallel reduction(+: a[0:2])
  { }

#pragma omp parallel reduction(+: a[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                               unsupported-note {{variable length arrays are not supported for the current target}}

  { }
}
#pragma omp end declare target

template <typename T>
void target_template(int arg) { // expected-note 2{{declared here}}
#pragma omp target
  {
    T vla[arg]; // unsupported-error {{variable length arrays are not supported for the current target}} \
                   expected-warning 2{{variable length arrays are a Clang extension}} \
                   expected-note 2{{function parameter 'arg' with unknown value cannot be used in a constant expression}}
  }
}

void target(int arg) { // expected-note 2{{declared here}}
#pragma omp target
  {
    int vla[arg]; // unsupported-error {{variable length arrays are not supported for the current target}} \
                     expected-warning {{variable length arrays are a Clang extension}} \
                     expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}
  }

#pragma omp target
  {
#pragma omp parallel
    {
      int vla[arg]; // unsupported-error {{variable length arrays are not supported for the current target}} \
                       expected-warning {{variable length arrays are a Clang extension}} \
                       expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}
    }
  }

  target_template<long>(arg); // expected-note {{in instantiation of function template specialization 'target_template<long>' requested here}}
}

void teams_reduction(int arg) { // expected-note {{declared here}}
  int a[2];
  int vla[arg];  // expected-warning {{variable length arrays are a Clang extension}} \
                    expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}

#pragma omp target map(a)
#pragma omp teams reduction(+: a)
  { }

#pragma omp target map(vla)
#pragma omp teams reduction(+: vla) // unsupported-error {{cannot generate code for reduction on variable length array}} \
                                       unsupported-note {{variable length arrays are not supported for the current target}}

  { }

#pragma omp target map(a[0:2])
#pragma omp teams reduction(+: a[0:2])
  { }

#pragma omp target map(vla[0:2])
#pragma omp teams reduction(+: vla[0:2])
  { }

#pragma omp target map(a[0:arg])
#pragma omp teams reduction(+: a[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                            unsupported-note {{variable length arrays are not supported for the current target}}

  { }

#pragma omp target map(vla[0:arg])
#pragma omp teams reduction(+: vla[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                              unsupported-note {{variable length arrays are not supported for the current target}}

  { }
}

void parallel_reduction(int arg) { // expected-note {{declared here}}
  int a[2];
  int vla[arg]; // expected-warning {{variable length arrays are a Clang extension}} \
                   expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}

#pragma omp target map(a)
#pragma omp parallel reduction(+: a)
  { }

#pragma omp target map(vla)
#pragma omp parallel reduction(+: vla) // unsupported-error {{cannot generate code for reduction on variable length array}} \
                                          unsupported-note {{variable length arrays are not supported for the current target}}

  { }

#pragma omp target map(a[0:2])
#pragma omp parallel reduction(+: a[0:2])
  { }

#pragma omp target map(vla[0:2])
#pragma omp parallel reduction(+: vla[0:2])
  { }

#pragma omp target map(a[0:arg])
#pragma omp parallel reduction(+: a[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                               unsupported-note {{variable length arrays are not supported for the current target}}

  { }

#pragma omp target map(vla[0:arg])
#pragma omp parallel reduction(+: vla[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                                 unsupported-note {{variable length arrays are not supported for the current target}}

  { }
}

void for_reduction(int arg) { // expected-note {{declared here}}
  int a[2];
  int vla[arg]; // expected-warning {{variable length arrays are a Clang extension}} \
                   expected-note {{function parameter 'arg' with unknown value cannot be used in a constant expression}}

#pragma omp target map(a)
#pragma omp parallel
#pragma omp for reduction(+: a)
  for (int i = 0; i < arg; i++) ;

#pragma omp target map(vla)
#pragma omp parallel
#pragma omp for reduction(+: vla) // unsupported-error {{cannot generate code for reduction on variable length array}} \
                                     unsupported-note {{variable length arrays are not supported for the current target}}

  for (int i = 0; i < arg; i++) ;

#pragma omp target map(a[0:2])
#pragma omp parallel
#pragma omp for reduction(+: a[0:2])
  for (int i = 0; i < arg; i++) ;

#pragma omp target map(vla[0:2])
#pragma omp parallel
#pragma omp for reduction(+: vla[0:2])
  for (int i = 0; i < arg; i++) ;

#pragma omp target map(a[0:arg])
#pragma omp parallel
#pragma omp for reduction(+: a[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                          unsupported-note {{variable length arrays are not supported for the current target}}

  for (int i = 0; i < arg; i++) ;

#pragma omp target map(vla[0:arg])
#pragma omp parallel
#pragma omp for reduction(+: vla[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                            unsupported-note {{variable length arrays are not supported for the current target}}

  for (int i = 0; i < arg; i++) ;
#pragma omp target reduction(+ : vla[0:arg]) // unsupported-error {{cannot generate code for reduction on array section, which requires a variable length array}} \
                                                 unsupported-note {{variable length arrays are not supported for the current target}}

  for (int i = 0; i < arg; i++) ;
}
