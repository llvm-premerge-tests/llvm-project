#include <asm/hwcap.h>
#include <asm/mman.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

int setup_mte() {
  return prctl(PR_SET_TAGGED_ADDR_CTRL, PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC,
               0, 0, 0);
}

int main(int argc, char const *argv[]) {
  if (!(getauxval(AT_HWCAP2) & HWCAP2_MTE))
    return 1;

  if (setup_mte())
    return 1;

  return 0; // Set a break point here.
}
