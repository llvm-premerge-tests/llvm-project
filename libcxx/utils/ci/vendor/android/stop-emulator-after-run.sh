#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Run a command, then stop the emulator afterwards whether the command succeeds
# or fails. If a build times out, then the container might not be stopped.

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
"${@}"
ERR=${?}
"${THIS_DIR}/stop-emulator.sh"
exit "${ERR}"
