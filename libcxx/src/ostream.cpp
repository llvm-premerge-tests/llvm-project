//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__availability>
#include <__config>
#ifndef _LIBCPP_HAS_NO_FILESYSTEM
#  include <fstream>
#endif
#include <ostream>

#include "std_stream.h"

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_AVAILABILITY_PRINT _LIBCPP_EXPORTED_FROM_ABI FILE* __get_ostream_file(ostream& __os) {
  auto* __rdbuf = __os.rdbuf();
#ifndef _LIBCPP_HAS_NO_FILESYSTEM
  if (auto* __buffer = dynamic_cast<filebuf*>(__rdbuf))
    return __buffer->__file();
#endif

  if (auto* __buffer = dynamic_cast<__stdoutbuf<char>*>(__rdbuf))
    return __buffer->__file();

  return nullptr;
}

_LIBCPP_END_NAMESPACE_STD
