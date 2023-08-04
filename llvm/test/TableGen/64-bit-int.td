// RUN: llvm-tblgen -DVALID -print-records %s | FileCheck %s
// RUN: not llvm-tblgen -DOOR1 -print-records %s 2>&1 | FileCheck %s --check-prefix=CHECK-OOR
// RUN: not llvm-tblgen -DOOR2 -print-records %s 2>&1 | FileCheck %s --check-prefix=CHECK-OOR
// RUN: not llvm-tblgen -DOOR3 -print-records %s 2>&1 | FileCheck %s --check-prefix=CHECK-OOR

def {

#ifdef OOR1
  bits<64> Val = -9223372036854775809;
#endif

#ifdef OOR2
  bits<64> Val = 0b10000000000000000000000000000000000000000000000000000000000000000;
#endif

#ifdef OOR3
  bits<64> Val = 0x10000000000000000;
#endif
// CHECK-OOR: error: Number out of range

#ifdef VALID
  bits<64> BinVal = 0x8000000000000000;
  bits<64> HexVal = 0b1000000000000000000000000000000000000000000000000000000000000000;
  bits<64> DecVal1 = 9223372036854775808;
  bits<64> DecVal2 = 9223372036854775809;
  bits<64> DecVal3 = -9223372036854775808;
#endif
// CHECK: bits<64> BinVal = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
// CHECK: bits<64> HexVal = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
// CHECK: bits<64> DecVal1 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
// CHECK: bits<64> DecVal2 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
// CHECK: bits<64> DecVal3 = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
}
