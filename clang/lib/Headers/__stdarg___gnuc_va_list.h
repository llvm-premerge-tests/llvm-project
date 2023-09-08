/*===---- __stdarg___gnuc_va_list.h - Definition of __gnuc_va_list ---------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* See stdarg.h for an explanation of builtin_headers_in_system_modules. */
#if !defined(__GNUC_VA_LIST) || __has_feature(builtin_headers_in_system_modules)
#define __GNUC_VA_LIST
typedef __builtin_va_list __gnuc_va_list;
#endif
