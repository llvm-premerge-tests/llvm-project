import argparse
import operator
import os.path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--libcxx-python-module', required=True)
parser.add_argument('--libcxx-include-directory', required=True)
args = parser.parse_args()

sys.path.append(args.libcxx_python_module)

import libcxx.header_information

public_headers = libcxx.header_information.public_headers
header_include_requirements = libcxx.header_information.header_include_requirements
always_available_headers = frozenset(public_headers).difference(*header_include_requirements.values())

with open(os.path.join(args.libcxx_include_directory, '__std_clang_module'), 'w') as std_clang_module_header:
    std_clang_module_header.write('''\
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header should not be directly included, it's exclusively to import all
// of the libc++ public clang modules for the `std` clang module to export. In
// other words, it's to facilitate `@import std;` in Objective-C++ and `import std`
// in Swift to expose all of the libc++ interfaces. This is generally not
// recommended, however there are some clients that need to import all of libc++
// without knowing what "all" is.
#if !__building_module(std)
#  error "Do not include this header directly, include individual headers instead"
#endif

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

''')
    for header in sorted(always_available_headers):
        std_clang_module_header.write('#include <')
        std_clang_module_header.write(header)
        std_clang_module_header.write('>\n')
    std_clang_module_header.write('\n')

    for requirements, headers in sorted(header_include_requirements.items(), key=operator.itemgetter(0)):
        if len(requirements) == 1:
            std_clang_module_header.write('#ifndef ')
            std_clang_module_header.write(requirements[0])
        else:
            std_clang_module_header.write('#if')
            for index, requirement in enumerate(requirements):
                if index > 0:
                    std_clang_module_header.write(' &&')
                std_clang_module_header.write(' !defined(')
                std_clang_module_header.write(requirement)
                std_clang_module_header.write(')')
        std_clang_module_header.write('\n')

        for header in sorted(headers):
            std_clang_module_header.write('#  include <')
            std_clang_module_header.write(header)
            std_clang_module_header.write('>\n')
        
        std_clang_module_header.write('#endif\n\n')
