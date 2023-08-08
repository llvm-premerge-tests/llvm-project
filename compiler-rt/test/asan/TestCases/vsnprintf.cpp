// RUN: %clangxx_asan -O2 %s -o %t
// RUN: %env_asan_opts=check_printf=1 %run %t 2>&1

// FIXME: printf is not intercepted on Windows yet.
// XFAIL: target={{.*windows.*}}

#include <stdarg.h>
#include <stdio.h>
#include <string>

void write(char *buf, int buf_size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, buf_size, fmt, args);
  va_end(args);
}

int main() {
  char buffer[100];
  std::string str(2147483648, '=');
  write(buffer, 100, "%s\n", str.c_str());
  printf("%s", buffer);
  return 0;
}
