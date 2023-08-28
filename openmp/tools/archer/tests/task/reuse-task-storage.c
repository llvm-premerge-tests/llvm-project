/*
 * reuse-task-storage.c -- Archer testcase
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//
// See tools/archer/LICENSE.txt for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


// RUN: %libarcher-compile-and-run | FileCheck %s
// REQUIRES: tsan
#include <omp.h>
#include <stdio.h>
#include <unistd.h>
#include "ompt/ompt-signal.h"

int main(int argc, char *argv[]) {
  int var = 0, a = 0;

#pragma omp parallel num_threads(8) shared(var, a)
  {
#pragma omp master
    for(int i=0; i<10; i++)
    {
      for(int j=0; j<100; j++)
#pragma omp task shared(var)
      {
        OMPT_SIGNAL(a);
        #pragma omp atomic update
          var++;
      }

      // Give other threads time to steal the task.
      OMPT_WAIT(a, (i*100));
    }

  }

  fprintf(stderr, "DONE\n");
  int error = (var != 1000);
  return error;
}

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: ThreadSanitizer: reported
// CHECK: DONE
