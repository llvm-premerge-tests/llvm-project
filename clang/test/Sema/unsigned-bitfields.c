// RUN: %clang_cc1 -fdump-record-layouts -emit-codegen-only %s -funsigned-bitfields | FileCheck %s

// CHECK: BitFields
// CHECK: Offset:0 Size:1 IsSigned:0
// CHECK: Offset:1 Size:1 IsSigned:1
// CHECK: Offset:2 Size:1 IsSigned:1
// CHECK: Offset:3 Size:1 IsSigned:1
// CHECK: Offset:4 Size:1 IsSigned:1
// CHECK: Offset:5 Size:1 IsSigned:1
// CHECK: Offset:6 Size:1 IsSigned:0
// CHECK: Offset:7 Size:1 IsSigned:0
// CHECK: Offset:8 Size:1 IsSigned:0
// CHECK: Offset:9 Size:1 IsSigned:0
// CHECK: Offset:10 Size:1 IsSigned:1
// CHECK: Offset:11 Size:1 IsSigned:1
// CHECK: Offset:12 Size:1 IsSigned:1
// CHECK: Offset:13 Size:1 IsSigned:1
// CHECK: Offset:14 Size:1 IsSigned:1
// CHECK: Offset:15 Size:1 IsSigned:0
// CHECK: Offset:16 Size:1 IsSigned:0
// CHECK: Offset:17 Size:1 IsSigned:0
// CHECK: Offset:18 Size:1 IsSigned:0
// CHECK: Offset:19 Size:1 IsSigned:1
// CHECK: Offset:20 Size:1 IsSigned:1
// CHECK: Offset:21 Size:1 IsSigned:1
// CHECK: Offset:22 Size:1 IsSigned:1
// CHECK: Offset:23 Size:1 IsSigned:1
// CHECK: Offset:24 Size:1 IsSigned:0
// CHECK: Offset:25 Size:1 IsSigned:0
// CHECK: Offset:26 Size:1 IsSigned:0
// CHECK: Offset:27 Size:1 IsSigned:0
// CHECK: Offset:28 Size:1 IsSigned:1
// CHECK: Offset:29 Size:1 IsSigned:1
// CHECK: Offset:30 Size:1 IsSigned:1
// CHECK: Offset:31 Size:1 IsSigned:1
// CHECK: Offset:32 Size:1 IsSigned:1
// CHECK: Offset:33 Size:1 IsSigned:0
// CHECK: Offset:34 Size:1 IsSigned:0
// CHECK: Offset:35 Size:1 IsSigned:0
// CHECK: Offset:36 Size:1 IsSigned:0
// CHECK: Offset:37 Size:1 IsSigned:1
// CHECK: Offset:38 Size:1 IsSigned:1
// CHECK: Offset:39 Size:1 IsSigned:1
// CHECK: Offset:40 Size:1 IsSigned:1
// CHECK: Offset:41 Size:1 IsSigned:1
// CHECK: Offset:42 Size:1 IsSigned:0
// CHECK: Offset:43 Size:1 IsSigned:0
// CHECK: Offset:44 Size:1 IsSigned:0



typedef char c;
typedef signed char sc;
typedef unsigned char uc;
typedef short s;
typedef signed short ss;
typedef unsigned short us;
typedef int i;
typedef signed int si;
typedef unsigned int ui;
typedef long l;
typedef signed long sl;
typedef unsigned long ul;
typedef long long ll;
typedef signed long long sll;
typedef unsigned long long ull;

typedef c ct;
typedef sc sct;
typedef uc uct;
typedef s st;
typedef ss sst;
typedef us ust;
typedef i it;
typedef si sit;
typedef ui uit;
typedef l lt;
typedef sl slt;
typedef ul ult;
typedef ll llt;
typedef sll sllt;
typedef ull ullt;

struct foo {
  char char0 : 1;
  c char1 : 1;
  ct char2 : 1;
  signed char schar0 : 1;
  sc schar1 : 1;
  sct schar2 : 1;
  unsigned char uchar0 : 1;
  uc uchar1 : 1;
  uct uchar2 : 1;
  short short0 : 1;
  s short1 : 1;
  st short2 : 1;
  signed short sshort0 : 1;
  ss sshort1 : 1;
  sst sshort2 : 1;
  unsigned short ushort0 : 1;
  us ushort1 : 1;
  ust ushort2 : 1;
  int int3 : 1;
  i int4 : 1;
  it int5 : 1;
  signed int sint0 : 1;
  si sint1 : 1;
  sit sint2 : 1;
  unsigned int uint0 : 1;
  ui uint1 : 1;
  uit uint2 : 1;
  long long0 : 1;
  l long1 : 1;
  lt long2 : 1;
  signed long slong0 : 1;
  sl slong1 : 1;
  slt slong2 : 1;
  unsigned long ulong0 : 1;
  ul ulong1 : 1;
  ult ulong2 : 1;
  long long llong0 : 1;
  ll llong1 : 1;
  llt llong2 : 1;
  signed long long sllong0 : 1;
  sll sllong1 : 1;
  sllt sllong2 : 1;
  unsigned long long ullong0 : 1;
  ull ullong1 : 1;
  ullt ullong2 : 1;
};

struct foo x;

extern void abort (void);
extern void exit (int);
extern void *memset (void *, int, __SIZE_TYPE__);

int
main (void)
{
  memset (&x, (unsigned char)-1, sizeof(x));
  if (x.char0 != 1 || x.char1 != 1 || x.char2 != 1
      || x.schar0 != -1 || x.schar1 != -1 || x.schar2 != -1
      || x.uchar0 != 1 || x.uchar1 != 1 || x.uchar2 != 1
      || x.short0 != 1 || x.short1 != 1 || x.short2 != 1
      || x.sshort0 != -1 || x.sshort1 != -1 || x.sshort2 != -1
      || x.ushort0 != 1 || x.ushort1 != 1 || x.ushort2 != 1
      || x.int3 != 1 || x.int4 != 1 || x.int5 != 1
      || x.sint0 != -1 || x.sint1 != -1 || x.sint2 != -1
      || x.uint0 != 1 || x.uint1 != 1 || x.uint2 != 1
      || x.long0 != 1 || x.long1 != 1 || x.long2 != 1
      || x.slong0 != -1 || x.slong1 != -1 || x.slong2 != -1
      || x.ulong0 != 1 || x.ulong1 != 1 || x.ulong2 != 1
      || x.llong0 != 1 || x.llong1 != 1 || x.llong2 != 1
      || x.sllong0 != -1 || x.sllong1 != -1 || x.sllong2 != -1
      || x.ullong0 != 1 || x.ullong1 != 1 || x.ullong2 != 1
      )
    abort();
  exit (0);
}
