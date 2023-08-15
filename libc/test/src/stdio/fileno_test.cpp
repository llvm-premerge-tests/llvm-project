//===-- Unittests for fileno ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "src/stdio/fileno.h"
#include "src/stdio/fopen.h"
#include "test/UnitTest/Test.h"
#include <stdio.h>

TEST(LlvmLibcFilenoTest, ValidFilenoTest) {
  ::FILE *file = __llvm_libc::fopen("testdata/test_data.txt", "w");
  ASSERT_FALSE(file == nullptr);

  // file descriptors associated with the streams stdin, stdout, and stderr are
  // 0, 1, and 2, respectively. Therefore any newly opened file should have an
  // integer file descriptor >= 3

  ASSERT_FALSE(__llvm_libc::fileno(file) < 3);
}

TEST(LlvmLibcFilenoTest, StandardStreamTest) {
  ASSERT_EQ(__llvm_libc::fileno(reinterpret_cast<FILE *>(__llvm_libc::stdin)),
            0);
  ASSERT_EQ(__llvm_libc::fileno(reinterpret_cast<FILE *>(__llvm_libc::stdout)),
            1);
  ASSERT_EQ(__llvm_libc::fileno(reinterpret_cast<FILE *>(__llvm_libc::stderr)),
            2);
}
