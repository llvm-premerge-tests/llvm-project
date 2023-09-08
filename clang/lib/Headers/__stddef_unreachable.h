/*===---- __stddef_unreachable.h - Definition of unreachable ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/* See stddef.h for an explanation of builtin_headers_in_system_modules. */
#if !defined(unreachable) || __has_feature(builtin_headers_in_system_modules)
#define unreachable() __builtin_unreachable()
#endif
