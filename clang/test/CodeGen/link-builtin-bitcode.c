// Build two version of the bitcode library, one with a target-cpu set and one without
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx803 -DBITCODE -emit-llvm-bc -o %t-lib.bc %s
// RUN: %clang_cc1 -triple amdgcn-- -DBITCODE -emit-llvm-bc -o %t-lib.no-cpu.bc %s

// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx90a -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx90a -emit-llvm \
// RUN:   -mlink-builtin-bitcode %t-lib.bc -o - %t.bc | FileCheck %s --check-prefixes=COMMON,CPU

// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx90a -emit-llvm-bc -o %t.bc %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx90a -emit-llvm \
// RUN:   -mlink-builtin-bitcode %t-lib.no-cpu.bc -o - %t.bc | FileCheck %s --check-prefixes=COMMON,NOCPU

#ifdef BITCODE
int no_attr(void) { return 42; }
int __attribute__((target("gfx8-insts"))) attr_in_target(void) { return 42; }
int __attribute__((target("extended-image-insts"))) attr_not_in_target(void) { return 42; }
int __attribute__((target("no-gfx9-insts"))) attr_uncompatible(void) { return 42; }
int x = 12;
#endif

extern int no_attr(void);
extern int attr_in_target(void);
extern int attr_not_in_target(void);
extern int attr_uncompatible(void);
extern int x;

int bar() { return no_attr() + attr_in_target() + attr_not_in_target() + attr_uncompatible() + x; }

// COMMON: @x = internal addrspace(1) global i32 12, align 4

// COMMON-LABEL: define dso_local i32 @bar
// COMMON-SAME: () #[[ATTR_BAR:[0-9]+]] {
//
// COMMON-LABEL: define internal i32 @no_attr
// COMMON-SAME: () #[[ATTR_COMPATIBLE:[0-9]+]] {

// COMMON-LABEL: define internal i32 @attr_in_target
// COMMON-SAME: () #[[ATTR_COMPATIBLE:[0-9]+]] {

// COMMON-LABEL: define internal i32 @attr_not_in_target
// COMMON-SAME: () #[[ATTR_EXTEND:[0-9]+]] {

// COMMON-LABEL: define internal i32 @attr_uncompatible
// COMMON-SAME: () #[[ATTR_UNCOMPATIBLE:[0-9]+]] {

// COMMON: attributes #[[ATTR_BAR]] = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
// COMMON: attributes #[[ATTR_COMPATIBLE]] = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gws,+image-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
// COMMON: attributes #[[ATTR_EXTEND]] = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-fadd-rtn-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gws,+image-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+extended-image-insts" }
// CPU: attributes #[[ATTR_UNCOMPATIBLE]] = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="+16-bit-insts,+ci-insts,+dpp,+gfx8-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,-gfx9-insts" }
// NOCPU: attributes #[[ATTR_UNCOMPATIBLE]] = { convergent noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx90a" "target-features"="-gfx9-insts" }
