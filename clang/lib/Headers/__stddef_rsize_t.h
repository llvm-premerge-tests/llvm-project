/*===---- __stddef_rsize_t.h - Definition of rsize_t -----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#if !defined(_RSIZE_T) || __has_feature(builtin_headers_in_system_modules)
#define _RSIZE_T

typedef __SIZE_TYPE__ rsize_t;

#endif
