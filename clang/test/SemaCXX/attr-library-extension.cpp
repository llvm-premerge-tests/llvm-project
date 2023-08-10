// RUN: %clang_cc1 -Wno-unused-value -std=c++03 -verify=cxx03,cxx11,cxx14,cxx17,cxx20,cxx23,gnu -fsyntax-only %s -DGNUAttr
// RUN: %clang_cc1 -Wno-unused-value -std=c++11 -verify=cxx11,cxx14,cxx17,cxx20,cxx23,gnu -fsyntax-only %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++14 -verify=cxx14,cxx17,cxx20,cxx23,gnu -fsyntax-only %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++17 -verify=cxx17,cxx20,cxx23,gnu -fsyntax-only %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -verify=cxx20,cxx23,gnu -fsyntax-only %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++23 -verify=cxx23,gnu -fsyntax-only %s
// RUN: %clang_cc1 -Wno-unused-value -std=c++26 -verify=gnu -fsyntax-only %s

#ifdef GNUAttr
#define EXTENSION(name) __attribute__((library_extension(name)))
#else
#define EXTENSION(name) [[clang::library_extension(name)]]
#endif

struct EXTENSION("C++11") StructCxx11Ext {}; // cxx03-note {{'StructCxx11Ext' has been explicitly marked as an extension here}}
struct EXTENSION("C++14") StructCxx14Ext {}; // cxx11-note {{'StructCxx14Ext' has been explicitly marked as an extension here}}
struct EXTENSION("C++17") StructCxx17Ext {}; // cxx14-note {{'StructCxx17Ext' has been explicitly marked as an extension here}}
struct EXTENSION("C++20") StructCxx20Ext {}; // cxx17-note {{'StructCxx20Ext' has been explicitly marked as an extension here}}
struct EXTENSION("C++23") StructCxx23Ext {}; // cxx20-note {{'StructCxx23Ext' has been explicitly marked as an extension here}}
struct EXTENSION("C++26") StructCxx26Ext {}; // cxx23-note {{'StructCxx26Ext' has been explicitly marked as an extension here}}
struct EXTENSION("GNU") GNUExt {}; // gnu-note {{'GNUExt' has been explicitly marked as an extension here}}

void consume(StructCxx11Ext); // cxx03-warning {{'StructCxx11Ext' is a C++11 extension}}
void consume(StructCxx14Ext); // cxx11-warning {{'StructCxx14Ext' is a C++14 extension}}
void consume(StructCxx17Ext); // cxx14-warning {{'StructCxx17Ext' is a C++17 extension}}
void consume(StructCxx20Ext); // cxx17-warning {{'StructCxx20Ext' is a C++20 extension}}
void consume(StructCxx23Ext); // cxx20-warning {{'StructCxx23Ext' is a C++23 extension}}
void consume(StructCxx26Ext); // cxx23-warning {{'StructCxx26Ext' is a C++2c extension}}
void consume(GNUExt); // gnu-warning {{'GNUExt' is a GNU extension}}

namespace EXTENSION("C++11") NSCxx11Ext { // cxx03-note {{'NSCxx11Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx11Ext::S); // cxx03-warning {{'NSCxx11Ext' is a C++11 extension}}

namespace EXTENSION("C++14") NSCxx14Ext { // cxx11-note {{'NSCxx14Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx14Ext::S); // cxx11-warning {{'NSCxx14Ext' is a C++14 extension}}

namespace EXTENSION("C++17") NSCxx17Ext { // cxx14-note {{'NSCxx17Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx17Ext::S); // cxx14-warning {{'NSCxx17Ext' is a C++17 extension}}

namespace EXTENSION("C++20") NSCxx20Ext { // cxx17-note {{'NSCxx20Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx20Ext::S); // cxx17-warning {{'NSCxx20Ext' is a C++20 extension}}

namespace EXTENSION("C++23") NSCxx23Ext { // cxx20-note {{'NSCxx23Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx23Ext::S); // cxx20-warning {{'NSCxx23Ext' is a C++23 extension}}

namespace EXTENSION("C++26") NSCxx26Ext { // cxx23-note {{'NSCxx26Ext' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSCxx26Ext::S); // cxx23-warning {{'NSCxx26Ext' is a C++2c extension}}

namespace EXTENSION("GNU") NSGNUExt { // gnu-note {{'NSGNUExt' has been explicitly marked as an extension here}}
  struct S {};
}
void consume(NSGNUExt::S); // gnu-warning {{'NSGNUExt' is a GNU extension}}

EXTENSION("C++11") void fcxx11(); // cxx03-note {{'fcxx11' has been explicitly marked as an extension here}}
EXTENSION("C++14") void fcxx14(); // cxx11-note {{'fcxx14' has been explicitly marked as an extension here}}
EXTENSION("C++17") void fcxx17(); // cxx14-note {{'fcxx17' has been explicitly marked as an extension here}}
EXTENSION("C++20") void fcxx20(); // cxx17-note {{'fcxx20' has been explicitly marked as an extension here}}
EXTENSION("C++23") void fcxx23(); // cxx20-note {{'fcxx23' has been explicitly marked as an extension here}}
EXTENSION("C++26") void fcxx26(); // cxx23-note {{'fcxx26' has been explicitly marked as an extension here}}
EXTENSION("GNU") void fgnu(); // gnu-note {{'fgnu' has been explicitly marked as an extension here}}

void call() {
  fcxx11(); // cxx03-warning {{'fcxx11' is a C++11 extension}}
  fcxx14(); // cxx11-warning {{'fcxx14' is a C++14 extension}}
  fcxx17(); // cxx14-warning {{'fcxx17' is a C++17 extension}}
  fcxx20(); // cxx17-warning {{'fcxx20' is a C++20 extension}}
  fcxx23(); // cxx20-warning {{'fcxx23' is a C++23 extension}}
  fcxx26(); // cxx23-warning {{'fcxx26' is a C++2c extension}}
  fgnu(); // gnu-warning {{'fgnu' is a GNU extension}}
}

EXTENSION("C++11") int vcxx11; // cxx03-note {{'vcxx11' has been explicitly marked as an extension here}}
EXTENSION("C++14") int vcxx14; // cxx11-note {{'vcxx14' has been explicitly marked as an extension here}}
EXTENSION("C++17") int vcxx17; // cxx14-note {{'vcxx17' has been explicitly marked as an extension here}}
EXTENSION("C++20") int vcxx20; // cxx17-note {{'vcxx20' has been explicitly marked as an extension here}}
EXTENSION("C++23") int vcxx23; // cxx20-note {{'vcxx23' has been explicitly marked as an extension here}}
EXTENSION("C++26") int vcxx26; // cxx23-note {{'vcxx26' has been explicitly marked as an extension here}}
EXTENSION("GNU") int vgnu; // gnu-note {{'vgnu' has been explicitly marked as an extension here}}

void access() {
  vcxx11; // cxx03-warning {{'vcxx11' is a C++11 extension}}
  vcxx14; // cxx11-warning {{'vcxx14' is a C++14 extension}}
  vcxx17; // cxx14-warning {{'vcxx17' is a C++17 extension}}
  vcxx20; // cxx17-warning {{'vcxx20' is a C++20 extension}}
  vcxx23; // cxx20-warning {{'vcxx23' is a C++23 extension}}
  vcxx26; // cxx23-warning {{'vcxx26' is a C++2c extension}}
  vgnu; // gnu-warning {{'vgnu' is a GNU extension}}
}

template <class>
class EXTENSION("C++11") TemplateCxx11Ext {}; // cxx03-note {{'TemplateCxx11Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx11Ext<void>); // cxx03-warning {{'TemplateCxx11Ext<void>' is a C++11 extension}}

template <class>
class EXTENSION("C++14") TemplateCxx14Ext {}; // cxx11-note {{'TemplateCxx14Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx14Ext<void>); // cxx11-warning {{'TemplateCxx14Ext<void>' is a C++14 extension}}

template <class>
class EXTENSION("C++17") TemplateCxx17Ext {}; // cxx14-note {{'TemplateCxx17Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx17Ext<void>); // cxx14-warning {{'TemplateCxx17Ext<void>' is a C++17 extension}}

template <class>
class EXTENSION("C++20") TemplateCxx20Ext {}; // cxx17-note {{'TemplateCxx20Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx20Ext<void>); // cxx17-warning {{'TemplateCxx20Ext<void>' is a C++20 extension}}

template <class>
class EXTENSION("C++23") TemplateCxx23Ext {}; // cxx20-note {{'TemplateCxx23Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx23Ext<void>); // cxx20-warning {{'TemplateCxx23Ext<void>' is a C++23 extension}}

template <class>
class EXTENSION("C++26") TemplateCxx26Ext {}; // cxx23-note {{'TemplateCxx26Ext<void>' has been explicitly marked as an extension here}}

void consume(TemplateCxx26Ext<void>); // cxx23-warning {{'TemplateCxx26Ext<void>' is a C++2c extension}}
