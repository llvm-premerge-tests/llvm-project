// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=4 -DUSER -x hip -o %t_4 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=5 -DUSER -x hip -o %t_5 %s

// RUN: %clang_cc1 -fcuda-is-device -triple amdgcn-amd-amdhsa -emit-llvm \
// RUN:   -mcode-object-version=none -DDEVICELIB -x hip -o %t_0 %s

// RUN: llvm-link %t_0 %t_4 -o -| llvm-dis -o - | FileCheck -check-prefix=LINKED4 %s
// RUN: llvm-link %t_0 %t_5 -o -| llvm-dis -o - | FileCheck -check-prefix=LINKED5 %s

#include "Inputs/cuda.h"

// LINKED4: llvm.amdgcn.abi.version = weak_odr hidden addrspace(4) constant i32 400, align 4
// LINKED4-LABEL: bar
// LINKED4: load i32, ptr addrspacecast (ptr addrspace(4) @llvm.amdgcn.abi.version to ptr), align {{.*}}
// LINKED4: [[ABI5_X:%.*]] = icmp sge i32 %{{.*}}, 500
// LINKED4: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED4: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// LINKED4: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED4: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// LINKED4: select i1 [[ABI5_X]], ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// LINKED4: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// LINKED4: "amdgpu_code_object_version", i32 400

// LINKED5: llvm.amdgcn.abi.version = weak_odr hidden addrspace(4) constant i32 500, align 4
// LINKED5-LABEL: bar
// LINKED5: load i32, ptr addrspacecast (ptr addrspace(4) @llvm.amdgcn.abi.version to ptr), align {{.*}}
// LINKED5: [[ABI5_X:%.*]] = icmp sge i32 %{{.*}}, 500
// LINKED5: call align 8 dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
// LINKED5: [[GEP_5_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 12
// LINKED5: call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// LINKED5: [[GEP_4_X:%.*]] = getelementptr i8, ptr addrspace(4) %{{.*}}, i32 4
// LINKED5: select i1 [[ABI5_X]], ptr addrspace(4) [[GEP_5_X]], ptr addrspace(4) [[GEP_4_X]]
// LINKED5: load i16, ptr addrspace(4) %{{.*}}, align 2, !range [[$WS_RANGE:![0-9]*]], !invariant.load{{.*}}, !noundef
// LINKED5: "amdgpu_code_object_version", i32 500

#ifdef DEVICELIB
__device__ void bar(int *out)
{
  *out = __builtin_amdgcn_workgroup_size_x();
}
#endif

#ifdef USER
__device__ void bar(int *out);
__device__ void foo()
{
  int *out;
  bar(out);
}
#endif