// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple aarch64-arm-none-eabi %s -o - | FileCheck %s

struct S0 {
  unsigned int x : 16;
  unsigned int y : 16;
};

struct S1 {
  volatile unsigned int x : 16;
  volatile unsigned int y : 16;
};

struct S2 {
  unsigned int x : 8;
  unsigned int y : 8;
};

struct S3 {
  volatile unsigned int x : 8;
  volatile unsigned int y : 8;
};

struct S4 {
  unsigned int x : 8;
  unsigned int y : 16;
};

struct S5 {
  volatile unsigned int x : 8;
  volatile unsigned int y : 16;
};

struct S6 {
  unsigned int x : 16;
  unsigned int y : 8;
};

struct S7 {
  volatile unsigned int x : 16;
  volatile unsigned int y : 8;
};

struct S8 {
  unsigned int x : 16;
  volatile unsigned int y : 16;
};

struct S9 {
  unsigned int x : 16;
  unsigned int y : 32;
};

struct S10 {
  const unsigned int x : 8;
  const volatile unsigned int y : 8;
  S10() : x(0), y(0) {}
};

// It's currently not possible to produce complete debug information for the following cases.
// Confirm that no wrong debug info is output.
// Once this is implemented, these tests should be amended.
struct S11 {
  unsigned int x : 15;
  unsigned int y : 16;
};

struct S12 {
  unsigned int x : 16;
  unsigned int y : 17;
};

struct __attribute__((packed)) S13 {
  unsigned int x : 15;
  unsigned int y : 16;
};

int main() {
// CHECK: %s0 = alloca %struct.S0, align 4
// CHECK: %s1 = alloca %struct.S1, align 4
// CHECK: %s2 = alloca %struct.S2, align 4
// CHECK: %s3 = alloca %struct.S3, align 4
// CHECK: %s4 = alloca %struct.S4, align 4
// CHECK: %s5 = alloca %struct.S5, align 4
// CHECK: %s6 = alloca %struct.S6, align 4
// CHECK: %s7 = alloca %struct.S7, align 4
// CHECK: %s8 = alloca %struct.S8, align 4
// CHECK: %s9 = alloca %struct.S9, align 4
// CHECK: %s10 = alloca %struct.S10, align 4
// CHECK: [[ADDR0:%.*]] = alloca %struct.S0, align 4
// CHECK: [[ADDR1:%.*]] = alloca %struct.S1, align 4
// CHECK: [[ADDR2:%.*]] = alloca %struct.S2, align 4
// CHECK: [[ADDR3:%.*]] = alloca %struct.S3, align 4
// CHECK: [[ADDR4:%.*]] = alloca %struct.S4, align 4
// CHECK: [[ADDR5:%.*]] = alloca %struct.S5, align 4
// CHECK: [[ADDR6:%.*]] = alloca %struct.S6, align 4
// CHECK: [[ADDR7:%.*]] = alloca %struct.S7, align 4
// CHECK: [[ADDR8:%.*]] = alloca %struct.S8, align 4
// CHECK: [[ADDR9:%.*]] = alloca %struct.S9, align 4
// CHECK: [[ADDR10:%.*]] = alloca %struct.S10, align 4
// CHECK: [[ADDR11:%.*]] = alloca %struct.S11, align 4
// CHECK: [[ADDR12:%.*]] = alloca %struct.S12, align 4
// CHECK: [[ADDR13:%.*]] = alloca %struct.S13, align 1
  S0 s0;
  S1 s1;
  S2 s2;
  S3 s3;
  S4 s4;
  S5 s5;
  S6 s6;
  S7 s7;
  S8 s8;
  S9 s9;
  S10 s10;
  S11 s11;
  S12 s12;
  S13 s13;

// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR0]], metadata [[A0:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR0]], metadata [[B0:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
  auto [a0, b0] = s0;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR1]], metadata [[A1:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR1]], metadata [[B1:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
  auto [a1, b1] = s1;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR2]], metadata [[A2:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR2]], metadata [[B2:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
  auto [a2, b2] = s2;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR3]], metadata [[A3:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR3]], metadata [[B3:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
  auto [a3, b3] = s3;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR4]], metadata [[A4:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR4]], metadata [[B4:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
  auto [a4, b4] = s4;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR5]], metadata [[A5:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR5]], metadata [[B5:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
  auto [a5, b5] = s5;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR6]], metadata [[A6:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR6]], metadata [[B6:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
  auto [a6, b6] = s6;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR7]], metadata [[A7:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR7]], metadata [[B7:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
  auto [a7, b7] = s7;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR8]], metadata [[A8:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR8]], metadata [[B8:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
  auto [a8, b8] = s8;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR9]], metadata [[A9:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR9]], metadata [[B9:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 4))
  auto [a9, b9] = s9;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR10]], metadata [[A10:![0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR10]], metadata [[B10:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
  auto [a10, b10] = s10;

  auto [a11, b11] = s11;
// CHECK: call void @llvm.dbg.declare(metadata ptr [[ADDR12]], metadata [[A12:![0-9]+]], metadata !DIExpression())
  auto [a12, b12] = s12;

  auto [a13, b13] = s13;
}

// CHECK: [[UINTTY:![0-9]+]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)

// CHECK: [[A0]] = !DILocalVariable(name: "a0", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY:![0-9]+]])
// CHECK: [[USHORTTY]] = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
// CHECK: [[B0]] = !DILocalVariable(name: "b0", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY]])

// CHECK: [[A1]] = !DILocalVariable(name: "a1", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUSHORTTY:![0-9]+]])
// CHECK: [[VOLATILEUSHORTTY]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[USHORTTY]])
// CHECK: [[B1]] = !DILocalVariable(name: "b1", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUSHORTTY]])

// CHECK: [[A2]] = !DILocalVariable(name: "a2", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHARTY:![0-9]+]])
// CHECK: [[UCHARTY]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
// CHECK: [[B2]] = !DILocalVariable(name: "b2", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHARTY]])

// CHECK: [[A3]] = !DILocalVariable(name: "a3", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUCHARTY:![0-9]+]])
// CHECK: [[VOLATILEUCHARTY]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[UCHARTY]])
// CHECK: [[B3]] = !DILocalVariable(name: "b3", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUCHARTY]])

// CHECK: [[A4]] = !DILocalVariable(name: "a4", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHARTY]])
// CHECK: [[B4]] = !DILocalVariable(name: "b4", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY]])

// CHECK: [[A5]] = !DILocalVariable(name: "a5", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUCHARTY]])
// CHECK: [[B5]] = !DILocalVariable(name: "b5", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUSHORTTY]])

// CHECK: [[A6]] = !DILocalVariable(name: "a6", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY]])
// CHECK: [[B6]] = !DILocalVariable(name: "b6", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHARTY]])

// CHECK: [[A7]] = !DILocalVariable(name: "a7", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUSHORTTY]])
// CHECK: [[B7]] = !DILocalVariable(name: "b7", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUCHARTY]])

// CHECK: [[A8]] = !DILocalVariable(name: "a8", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY]])
// CHECK: [[B8]] = !DILocalVariable(name: "b8", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILEUSHORTTY]])

// CHECK: [[A9]] = !DILocalVariable(name: "a9", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORTTY]])
// CHECK: [[B9]] = !DILocalVariable(name: "b9", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINTTY]])

// CHECK: [[A10]] = !DILocalVariable(name: "a10", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONSTUCHARTY:![0-9]+]])
// CHECK: [[CONSTUCHARTY]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[UCHARTY]])
// CHECK: [[B10]] = !DILocalVariable(name: "b10", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONSTVOLATILEUCHARTY:![0-9]+]])
// CHECK: [[CONSTVOLATILEUCHARTY]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[VOLATILEUCHARTY]])

// CHECK-NOT: !DILocalVariable(name: "a11")
// CHECK-NOT: !DILocalVariable(name: "b11")

// CHECK: [[A12]] = !DILocalVariable(name: "a12", scope: {{.*}}, file: {{.*}}, type: [[USHORTTY]])
// CHECK-NOT: !DILocalVariable(name: "b12")

// CHECK-NOT: !DILocalVariable(name: "a13")
// CHECK-NOT: !DILocalVariable(name: "b13")
