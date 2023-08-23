// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=alpha.security.cert.env.InvalidPtr \
// RUN:  -analyzer-config alpha.security.cert.env.InvalidPtr:InvalidatingGetEnv=false \
// RUN:  -analyzer-output=text -verify -Wno-unused %s
//
// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=alpha.security.cert.env.InvalidPtr \
// RUN:  -analyzer-config \
// RUN: alpha.security.cert.env.InvalidPtr:InvalidatingGetEnv=true \
// RUN: -analyzer-output=text -verify=pedantic -Wno-unused %s

#include "Inputs/system-header-simulator.h"

char *getenv(const char *name);
int setenv(const char *name, const char *value, int overwrite);
int strcmp(const char *, const char *);

int custom_env_handler(const char **envp);

void getenv_after_getenv(void) {
  char *v1 = getenv("V1");
  // pedantic-note@-1{{previous function call was here}}

  char *v2 = getenv("V2");
  // pedantic-note@-1{{'getenv' call may invalidate the result of the previous 'getenv'}}

  strcmp(v1, v2);
  // pedantic-warning@-1{{use of invalidated pointer 'v1' in a function call}}
  // pedantic-note@-2{{use of invalidated pointer 'v1' in a function call}}
}

void setenv_after_getenv(void) {
  char *v1 = getenv("VAR1");

  setenv("VAR2", "...", 1);
  // expected-note@-1{{'setenv' call may invalidate the environment returned by getenv}}
  // pedantic-note@-2{{'setenv' call may invalidate the environment returned by getenv}}

  strcmp(v1, "");
  // expected-warning@-1{{use of invalidated pointer 'v1' in a function call}}
  // expected-note@-2{{use of invalidated pointer 'v1' in a function call}}
  // pedantic-warning@-3{{use of invalidated pointer 'v1' in a function call}}
  // pedantic-note@-4{{use of invalidated pointer 'v1' in a function call}}
}

int main(int argc, const char *argv[], const char *envp[]) {
  setenv("VAR", "...", 0);
  // expected-note@-1 2 {{'setenv' call may invalidate the environment parameter of 'main'}}
  // pedantic-note@-2 2 {{'setenv' call may invalidate the environment parameter of 'main'}}

  *envp;
  // expected-warning@-1 2 {{dereferencing an invalid pointer}}
  // expected-note@-2 2 {{dereferencing an invalid pointer}}
  // pedantic-warning@-3 2 {{dereferencing an invalid pointer}}
  // pedantic-note@-4 2 {{dereferencing an invalid pointer}}
}
