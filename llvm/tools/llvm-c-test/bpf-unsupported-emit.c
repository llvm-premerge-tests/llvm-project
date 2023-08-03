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
  const char *CSeverity = "unhandled";
  switch (LLVMGetDiagInfoSeverity(DI)) {
  case LLVMDSError:
    CSeverity = "error";
    break;
  case LLVMDSWarning:
    CSeverity = "warning";
    break;
  case LLVMDSRemark:
    CSeverity = "remark";
    break;
  case LLVMDSNote:
    CSeverity = "note";
    break;
  }
  char *CErr = LLVMGetDiagInfoDescription(DI);
  fprintf(stderr, "%s: %s\n", CSeverity, CErr);
  LLVMDisposeMessage(CErr);
}

int llvm_test_bpf_unsupported_emit(void) {
  LLVMInitializeBPFTargetInfo();
  LLVMInitializeBPFTarget();
  LLVMInitializeBPFTargetMC();
  LLVMInitializeBPFAsmPrinter();

  const char *BPF = "bpf";

  LLVMTargetRef Target;
  {
    char *ErrorMessage = NULL;
    if (LLVMGetTargetFromTriple(BPF, &Target, &ErrorMessage)) {
      fprintf(stderr, "LLVMGetTargetFromTriple(%s) failed: %s\n", BPF,
              ErrorMessage);
      LLVMDisposeMessage(ErrorMessage);
      return EXIT_FAILURE;
    }
  }

  LLVMContextRef C = LLVMContextCreate();
  LLVMContextSetDiagnosticHandler(C, diagnosticHandler, NULL);

  LLVMMemoryBufferRef MemBuf;
  {
    char *ErrorMessage = NULL;
    if (LLVMCreateMemoryBufferWithSTDIN(&MemBuf, &ErrorMessage)) {
      fprintf(stderr, "LLVMCreateMemoryBufferWithSTDIN failed: %s\n",
              ErrorMessage);
      LLVMDisposeMessage(ErrorMessage);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  LLVMModuleRef M;
  {
    char *ErrorMessage = NULL;
    if (LLVMParseIRInContext(C, MemBuf, &M, &ErrorMessage)) {
      fprintf(stderr, "LLVMParseIRInContext failed: %s\n", ErrorMessage);
      LLVMDisposeMessage(ErrorMessage);
      LLVMDisposeMemoryBuffer(MemBuf);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  LLVMTargetMachineRef TM =
      LLVMCreateTargetMachine(Target, BPF, NULL, NULL, LLVMCodeGenLevelDefault,
                              LLVMRelocDefault, LLVMCodeModelDefault);

  LLVMMemoryBufferRef ObjectFileBuffer;
  {
    char *ErrorMessage = NULL;
    if (LLVMTargetMachineEmitToMemoryBuffer(TM, M, LLVMAssemblyFile,
                                            &ErrorMessage, &ObjectFileBuffer)) {
      fprintf(stderr, "LLVMTargetMachineEmitToMemoryBuffer failed: %s\n",
              ErrorMessage);
      LLVMDisposeMessage(ErrorMessage);
      LLVMDisposeTargetMachine(TM);
      LLVMDisposeModule(M);
      LLVMContextDispose(C);
      return EXIT_FAILURE;
    }
  }

  int Ret = EXIT_SUCCESS;

  const char *Start = LLVMGetBufferStart(ObjectFileBuffer);
  const size_t Size = LLVMGetBufferSize(ObjectFileBuffer);
  for (size_t W = 0; W != Size;) {
    size_t N = fwrite(Start + W, 1, Size - W, stdout);
    if (N == 0) {
      fprintf(stderr, "fwrite failed after %zu/%zu bytes\n", W, Size);
      Ret = EXIT_FAILURE;
      break;
    }
    W += N;
  }

  LLVMDisposeMemoryBuffer(ObjectFileBuffer);
  LLVMDisposeModule(M);
  LLVMDisposeTargetMachine(TM);
  LLVMContextDispose(C);

  return Ret;
}
