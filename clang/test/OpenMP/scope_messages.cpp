// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 %s -verify=expected,omp51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 %s -verify=expected,omp51

void test1()
{
  int var1;
  int var2;
  int var3 = 1;

  // expected-error@+1 {{directive '#pragma omp scope' cannot contain more than one 'nowait' clause}} //omp51-error@+1{{unexpected OpenMP clause 'firstprivate' in directive '#pragma omp scope'}}
  #pragma omp scope private(var1) firstprivate(var3) nowait nowait
  { var1 = 123; ++var2; var3 = 2;}
}

void bar();

void test2() {
#pragma omp for
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'for' region}}
    bar();
  }

#pragma omp for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp sections
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'sections' region}}
    bar();
  }

#pragma omp sections
  {
#pragma omp section
    {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'section' region}}
      bar();
    }
  }

#pragma omp single
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'single' region}}
    bar();
  }

#pragma omp master
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'master' region}}
    bar();
  }

#pragma omp critical
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'critical' region}}
    bar();
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'parallel for' region}}
    bar();
  }

#pragma omp parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp parallel master
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'parallel master' region}}
    bar();
  }

#pragma omp parallel sections
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'parallel sections' region}}
    bar();
  }

#pragma omp task
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'task' region}}
    bar();
  }

#pragma omp ordered
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'ordered' region}}
    bar();
  }

#pragma omp atomic
  // expected-error@+2 {{the statement for 'atomic' must be an expression statement of form '++x;', '--x;', 'x++;', 'x--;', 'x binop= expr;', 'x = x binop expr' or 'x = expr binop x', where x is an lvalue expression with scalar type}}
  // expected-note@+1 {{expected an expression statement}}
  {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside an atomic region}}
    bar();
  }

#pragma omp target
#pragma omp teams
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'teams' region; perhaps you forget to enclose 'omp scope' directive into a parallel region?}}
    bar();
  }

#pragma omp taskloop
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'taskloop' region}}
    bar();
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'distribute parallel for' region}}
    bar();
  }

#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp target simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp target
#pragma omp teams distribute simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp target
#pragma omp teams distribute parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp target
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'teams distribute parallel for' region}}
    bar();
  }

#pragma omp target teams
  {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'target teams' region; perhaps you forget to enclose 'omp scope' directive into a parallel region?}}
    bar();
  }

#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{region cannot be closely nested inside 'target teams distribute parallel for' region}}
    bar();
  }

#pragma omp target teams distribute parallel for simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp target teams distribute simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }

#pragma omp simd
  for (int i = 0; i < 10; ++i) {
#pragma omp scope // expected-error {{OpenMP constructs may not be nested inside a simd region}}
    bar();
  }
}
