#!/bin/sh

# This script makes it easy to pass an image and arguments to qemu.

# Usage:
# run-qemu.sh QEMU_EXECUTABLE MACHINE CPU IMAGE ARGS...

QEMU_EXECUTABLE=$1; shift
MACHINE=$1; shift
CPU=$1; shift
IMAGE=$1; shift

semihosting_config="enable=on,chardev=stdio0"
for arg in "$@"
do
    semihosting_config="${semihosting_config},arg=${arg}"
done

"${QEMU_EXECUTABLE}" \
    -chardev stdio,mux=on,id=stdio0 \
    -semihosting-config "${semihosting_config}" \
    -monitor none \
    -serial none \
    -machine "${MACHINE},accel=tcg" \
    -cpu "${CPU}" \
    -device loader,file="${IMAGE}",cpu-num=0 \
    -nographic
