# REQUIRES: exegesis-can-measure-latency, x86_64-linux

# RUN: llvm-exegesis -mtriple=x86_64-unknown-unknown -mode=latency -snippets-file=%s -execution-mode=subprocess | FileCheck %s

# CHECK: error:           A segmentation fault occurred at address 20000

# LLVM-EXEGESIS-DEFREG RBX 20000
movq (%rbx), %rax
