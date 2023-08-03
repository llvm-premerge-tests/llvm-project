/*===-- targets.c - tool for testing libLLVM and llvm-c API ---------------===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file implements the --bpf-unsupported-emit command in llvm-c-test.    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c-test.h"
#include "llvm-c/IRReader.h"
#include "llvm-c/Target.h"
#include "llvm-c/TargetMachine.h"

#include <stdio.h>
#include <stdlib.h>

static void diagnosticHandler(LLVMDiagnosticInfoRef DI, void *C) {
  const char *severity = "unhandled";
  switch (LLVMGetDiagInfoSeverity(DI)) {
  case LLVMDSError:
    severity = "error";
    break;
  case LLVMDSWarning:
    severity = "warning";
    break;
  case LLVMDSRemark:
    severity = "remark";
    break;
  case LLVMDSNote:
    severity = "note";
    break;
  }
  char *CErr = LLVMGetDiagInfoDescription(DI);
  fprintf(stderr, "%s: %s\n", severity, CErr);
  LLVMDisposeMessage(CErr);
}

int llvm_test_bpf_unsupported_emit(void) {
  LLVMInitializeBPFTargetInfo();
  LLVMInitializeBPFTarget();
  LLVMInitializeBPFTargetMC();
  LLVMInitializeBPFAsmPrinter();

  const char *bpf = "bpf";

  LLVMTargetRef target;
  {
    char *error_message = NULL;
    if (LLVMGetTargetFromTriple(bpf, &target, &error_message)) {
      fprintf(stderr, "LLVMGetTargetFromTriple(%s) failed: %s\n", bpf,
              error_message);
      LLVMDisposeMessage(error_message);
      return EXIT_FAILURE;
    }
  }

  LLVMContextRef C = LLVMContextCreate();
  LLVMContextSetDiagnosticHandler(C, diagnosticHandler, NULL);

  LLVMMemoryBufferRef mem_buf;
  {
    char *error_message = NULL;
    if (LLVMCreateMemoryBufferWithSTDIN(&mem_buf, &error_message)) {
      fprintf(stderr, "LLVMCreateMemoryBufferWithSTDIN failed: %s\n",
              error_message);
      LLVMDisposeMessage(error_message);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  LLVMModuleRef mod;
  {
    char *error_message = NULL;
    if (LLVMParseIRInContext(C, mem_buf, &mod, &error_message)) {
      fprintf(stderr, "LLVMParseIRInContext failed: %s\n", error_message);
      LLVMDisposeMessage(error_message);
      LLVMDisposeMemoryBuffer(mem_buf);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  LLVMTargetMachineRef target_machine =
      LLVMCreateTargetMachine(target, bpf, NULL, NULL, LLVMCodeGenLevelDefault,
                              LLVMRelocDefault, LLVMCodeModelDefault);

  LLVMMemoryBufferRef asm_buf;
  {
    char *error_message = NULL;
    if (LLVMTargetMachineEmitToMemoryBuffer(
            target_machine, mod, LLVMAssemblyFile, &error_message, &asm_buf)) {
      fprintf(stderr, "LLVMTargetMachineEmitToMemoryBuffer failed: %s\n",
              error_message);
      LLVMDisposeMessage(error_message);
      LLVMDisposeTargetMachine(target_machine);
      LLVMDisposeModule(mod);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  int ret = EXIT_SUCCESS;

  const char *asm_start = LLVMGetBufferStart(asm_buf);
  const size_t asm_size = LLVMGetBufferSize(asm_buf);
  for (size_t wrote = 0; wrote != asm_size;) {
    size_t n = fwrite(asm_start, 1, asm_size - wrote, stdout);
    if (n == 0) {
      fprintf(stderr, "fwrite failed after %zu/%zu bytes\n", wrote, asm_size);
      ret = EXIT_FAILURE;
      break;
    }
    wrote += n;
  }

  LLVMDisposeMemoryBuffer(asm_buf);
  LLVMDisposeModule(mod);
  LLVMDisposeTargetMachine(target_machine);
  LLVMContextDispose(C);

  return ret;
}
