// RUN: %clang_analyze_cc1 -analyzer-checker=optin.mpi.MPI-Checker -verify %s

#include "MPIMock.h"

bool contains();
void do_a() {
  if (contains()) {
    MPI_Request request_item;
    MPI_Wait(&request_item, MPI_STATUS_IGNORE);
    // expected-warning@-1 {{Request 'request_item' has no matching nonblocking call.}}
  }
  do_a();
}
