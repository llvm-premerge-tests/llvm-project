/*===---- __stdarg_va_list.h - Definition of va_list -----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* See stdarg.h for an explanation of builtin_headers_in_system_modules. */
#if !defined(_VA_LIST) || __has_feature(builtin_headers_in_system_modules)
#define _VA_LIST
typedef __builtin_va_list va_list;
#endif
