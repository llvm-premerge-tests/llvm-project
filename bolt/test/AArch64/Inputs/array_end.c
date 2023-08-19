
__attribute__((destructor)) void destr() {}

__attribute__((noinline)) void callFini() {
  extern void (*__fini_array_start[])();
  extern void (*__fini_array_end[])();
  unsigned long Count = __fini_array_end - __fini_array_start;
  for (unsigned long I = 0; I < Count; ++I)
    (*__fini_array_start[I])();
}

void _start() {
  callFini();
  __asm__("mov     x0, #0\n"
          "mov     w8, #93\n"
          "svc     #0\n");
}
