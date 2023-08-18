#ifndef HEADER
#define HEADER
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify %s

#define NNN 50
int aaa[NNN];
int aaa2[NNN][NNN];

void parallel_loop() {
  #pragma omp parallel
  {
     #pragma omp loop
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }
   }
}

void parallel_for_AND_loop_bind() {
  #pragma omp parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void teams_loop() {
  int var1, var2;

  #pragma omp teams
  {
     #pragma omp loop bind(teams)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }

     #pragma omp loop bind(teams) collapse(2) private(var1)
     for (int i = 0 ; i < 3 ; i++) {
       for (int j = 0 ; j < NNN ; j++) {
         var1 += aaa[j];
       }
     }
   }
}

void orphan_loop_with_bind() {
  #pragma omp loop bind(parallel) 
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void orphan_loop_no_bind() {
  #pragma omp loop  // expected-error{{expected 'bind' clause for 'loop' construct without an enclosing OpenMP construct}}
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void teams_loop_reduction() {
  int total = 0;

  #pragma omp teams
  {
     #pragma omp loop bind(teams)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }

     #pragma omp loop bind(teams) reduction(+:total) // expected-error{{'reduction' clause not allowed with '#pragma omp loop bind(teams)'}}
     for (int j = 0 ; j < NNN ; j++) {
       total+=aaa[j];
     }
   }
}

void orphan_loop_teams_bind(){
  #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'unknown' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
  for (int i = 0; i < NNN; i++) {
    aaa[i] = i+i*NNN;
  }
}

void parallel_for_with_loop_teams_bind(){
  #pragma omp parallel for
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }
}

int main(int argc, char *argv[]) {
  parallel_loop();
  parallel_for_AND_loop_bind();
  teams_loop();
  orphan_loop_with_bind();
  orphan_loop_no_bind();
  teams_loop_reduction();
  orphan_loop_teams_bind();
  parallel_for_with_loop_teams_bind();
}

#endif
