/*===---- __stdarg_va_copy.h - Definition of va_copy------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* See stdarg.h for an explanation of builtin_headers_in_system_modules. */
#if !defined(va_copy) || __has_feature(builtin_headers_in_system_modules)
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#endif
