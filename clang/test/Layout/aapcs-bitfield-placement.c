// RUN: %clang_cc1 -triple armv8-none-linux-eabi   -fno-aapcs-bitfield-width -emit-llvm -o %t %s -fdump-record-layouts-simple | FileCheck %s -check-prefixes=LAYOUT,LAYOUT_LE
// RUN: %clang_cc1 -triple armebv8-none-linux-eabi   -fno-aapcs-bitfield-width -emit-llvm -o %t %s -fdump-record-layouts-simple | FileCheck %s -check-prefixes=LAYOUT,LAYOUT_BE

// RUN: %clang_cc1 -triple armv8-none-linux-eabi   -faapcs-bitfield-width -emit-llvm -o %t %s -fdump-record-layouts-simple | FileCheck %s -check-prefixes=LAYOUT,LAYOUT_LE
// RUN: %clang_cc1 -triple armebv8-none-linux-eabi   -faapcs-bitfield-width -emit-llvm -o %t %s -fdump-record-layouts-simple | FileCheck %s -check-prefixes=LAYOUT,LAYOUT_BE

struct st0 {
  short c : 7;
} st0;
// LAYOUT-LABEL: Record: {{.*}} struct st0 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st0 = type { i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st1 {
  int a : 10;
  short c : 6;
} st1;
// LAYOUT-LABEL: Record: {{.*}} struct st1 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st1 = type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:10 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:10 Size:6 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:6 Size:10 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:6 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st2 {
  int a : 10;
  short c : 7;
} st2;
// LAYOUT-LABEL: Record: {{.*}} struct st2 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st2 = type { i16, i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:10 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:2 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:6 Size:10 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:2 
// LAYOUT-NEXT: ]>

struct st3 {
  volatile short c : 7;
} st3;
// LAYOUT-LABEL: Record: {{.*}} struct st3 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st3 = type { i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:1 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st4 {
  int b : 9;
  volatile char c : 5;
} st4;
// LAYOUT-LABEL: Record: {{.*}} struct st4 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st4 = type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:9 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:9 Size:5 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:7 Size:9 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:2 Size:5 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st5 {
  int a : 12;
  volatile char c : 5;
} st5;
// LAYOUT-LABEL: Record: {{.*}} struct st5 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st5 = type { i16, i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:5 IsSigned:1 StorageSize:8 StorageOffset:2 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:3 Size:5 IsSigned:1 StorageSize:8 StorageOffset:2 
// LAYOUT-NEXT: ]>

struct st6 {
  int a : 12;
  char b;
  int c : 5;
} st6;
// LAYOUT-LABEL: Record: {{.*}} struct st6 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st6 = type { i16, i8, i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:12 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:5 IsSigned:1 StorageSize:8 StorageOffset:3 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:4 Size:12 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:3 Size:5 IsSigned:1 StorageSize:8 StorageOffset:3 
// LAYOUT-NEXT: ]>

struct st7a {
  char a;
  int b : 5;
} st7a;
// LAYOUT-LABEL: Record: {{.*}} struct st7a definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st7a = type { i8, i8, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:5 IsSigned:1 StorageSize:8 StorageOffset:1 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:3 Size:5 IsSigned:1 StorageSize:8 StorageOffset:1 
// LAYOUT-NEXT: ]>

struct st7b {
  char x;
  volatile struct st7a y;
} st7b;
// LAYOUT-LABEL: Record: {{.*}} struct st7b definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st7b = type { i8, [3 x i8], %struct.st7a }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: ]>

struct st8 {
  unsigned f : 16;
} st8;
// LAYOUT-LABEL: Record: {{.*}} struct st8 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st8 = type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:0 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:0 StorageSize:16 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st9{
  int f : 8;
} st9;
// LAYOUT-LABEL: Record: {{.*}} struct st9 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st9 = type { i8, [3 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st10{
  int e : 1;
  int f : 8;
} st10;
// LAYOUT-LABEL: Record: {{.*}} struct st10 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st10 = type { i16, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:1 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:1 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:15 Size:1 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:7 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st11{
  char e;
  int f : 16;
} st11;
// LAYOUT-LABEL: Record: {{.*}} struct st11 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st11 = type <{ i8, i16, i8 }>
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:16 StorageOffset:1 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:16 StorageOffset:1 
// LAYOUT-NEXT: ]>

struct st12{
  int e : 8;
  int f : 16;
} st12;
// LAYOUT-LABEL: Record: {{.*}} struct st12 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st12 = type { i24 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:32 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:32 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:24 Size:8 IsSigned:1 StorageSize:32 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:8 Size:16 IsSigned:1 StorageSize:32 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st13 {
  char a : 8;
  int b : 32;
} __attribute__((packed)) st13;
// LAYOUT-LABEL: Record: {{.*}} struct st13 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st13 = type { [5 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:8 Size:32 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st14 {
  char a : 8;
} __attribute__((packed)) st14;
// LAYOUT-LABEL: Record: {{.*}} struct st14 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st14 = type { i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st15 {
  short a : 8;
} __attribute__((packed)) st15;
// LAYOUT-LABEL: Record: {{.*}} struct st15 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st15 = type { i8 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct st16 {
  int a : 32;
  int b : 16;
  int c : 32;
  int d : 16;
} st16;
// LAYOUT-LABEL: Record: {{.*}} struct st16 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st16 = type { i48, i48 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:64 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:32 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:64 StorageOffset:8 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:32 Size:16 IsSigned:1 StorageSize:64 StorageOffset:8 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:32 Size:32 IsSigned:1 StorageSize:64 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:32 Size:32 IsSigned:1 StorageSize:64 StorageOffset:8 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:16 Size:16 IsSigned:1 StorageSize:64 StorageOffset:8 
// LAYOUT-NEXT: ]>

struct st17 {
int b : 32;
char c : 8;
} __attribute__((packed)) st17;
// LAYOUT-LABEL: Record: {{.*}} struct st17 definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.st17 = type { [5 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:32 Size:8 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:8 Size:32 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:40 StorageOffset:0 
// LAYOUT-NEXT: ]>

struct zero_bitfield {
  int a : 8;
  char : 0;
  int b : 8;
} st18;
// LAYOUT-LABEL: Record: {{.*}} struct zero_bitfield definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.zero_bitfield = type { i8, i8, [2 x i8] }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:1 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:1 
// LAYOUT-NEXT: ]>

struct zero_bitfield_ok {
  short a : 8;
  char a1 : 8;
  long : 0;
  int b : 24;
} st19;
// LAYOUT-LABEL: Record: {{.*}} struct zero_bitfield_ok definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.zero_bitfield_ok = type { i16, i24 }
// LAYOUT: BitFields:[
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_LE-NEXT: <CGBitFieldInfo Offset:0 Size:24 IsSigned:1 StorageSize:32 StorageOffset:4 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:16 StorageOffset:0 
// LAYOUT_BE-NEXT: <CGBitFieldInfo Offset:8 Size:24 IsSigned:1 StorageSize:32 StorageOffset:4 
// LAYOUT-NEXT: ]>


