// RUN: %libomp-compile && env OMP_NUM_THREADS=2 KMP_ENABLE_TASK_THROTTLING=1
// KMP_TASK_MAXIMUM=0      %libomp-run RUN: %libomp-compile && env
// OMP_NUM_THREADS=2 KMP_ENABLE_TASK_THROTTLING=1 KMP_TASK_MAXIMUM=1 %libomp-run
// RUN: %libomp-compile && env OMP_NUM_THREADS=2 KMP_ENABLE_TASK_THROTTLING=1
// KMP_TASK_MAXIMUM=256    %libomp-run RUN: %libomp-compile && env
// OMP_NUM_THREADS=2 KMP_ENABLE_TASK_THROTTLING=1 KMP_TASK_MAXIMUM=65536
// %libomp-run RUN: %libomp-compile && env OMP_NUM_THREADS=2
// KMP_ENABLE_TASK_THROTTLING=1 KMP_TASK_MAXIMUM=100000 %libomp-run

/**
 *  This test ensures that task throttling on the maximum number of tasks
 *  threshold works properly.
 *
 *  It creates 2 threads (1 producer, 1 consummer)
 *  The producer infinitely create tasks 'T_i' until one executed
 *  The consumer is blocked until the producer starts throttling
 *  Executing any 'T_i' unblocks the consumer and stop the producer
 *
 *  The assertion tests ensures that the producer does not create more than the
 *  total number of tasks provided by the programmer
 */

#include <assert.h>
#include <omp.h>
#include <stdlib.h>

/* default value */
#define MAX_TASKS_DEFAULT (65536)

int main(void) {
  /* maximum number of tasks in-flight */
  char *max_tasks_str = getenv("KMP_TASK_MAXIMUM");
  int max_tasks = max_tasks_str ? atoi(max_tasks_str) : MAX_TASKS_DEFAULT;
  if (max_tasks <= 0)
    max_tasks = 1;

  /* check if throttling is enabled (it is by default) */
  char *throttling_str = getenv("KMP_ENABLE_TASK_THROTTLING");
  int throttling = throttling_str ? *throttling_str == '1' : 1;
  assert(throttling);

  volatile int done = 0;

/* testing KMP_TASK_MAXIMUM */
#pragma omp parallel num_threads(2) default(none)                              \
    shared(max_tasks, throttling, done)
  {
    if (omp_get_thread_num() == 1)
      while (!done)
        ;

#pragma omp master
    {
      int ntasks = 0;
      while (!done) {
#pragma omp task default(none) shared(done) depend(out : max_tasks, throttling)
        done = 1;

        assert(++ntasks <= max_tasks + 1);
      }
    }
  }

  return 0;
}
