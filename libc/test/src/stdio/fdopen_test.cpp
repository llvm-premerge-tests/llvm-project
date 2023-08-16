//===-- Unittests for fopen / fclose --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"
#include "src/errno/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fdopen.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

#include "test/UnitTest/Test.h"

TEST(LlvmLibcFdOpenTest, TestOpen) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  int fd = __llvm_libc::open("./testdata/test_data.txt", O_WRONLY | O_CREAT,
                             S_IRWXU);

  ASSERT_EQ(libc_errno, 0);
  ASSERT_GT(fd, 2);

  ::FILE *file = __llvm_libc::fdopen(fd, "r");
  ASSERT_FALSE(file == nullptr);
  ASSERT_THAT(__llvm_libc::fclose(file), Succeeds(0));
}
